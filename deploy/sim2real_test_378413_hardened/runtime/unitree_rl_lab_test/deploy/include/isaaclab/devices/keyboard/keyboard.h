#pragma once

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/select.h>
#include <termios.h>
#include <thread>
#include <unistd.h>

// Terminal keyboard reader with a bounded event queue. The queue prevents the
// fast FSM loop from consuming a key edge before the 50 Hz policy loop sees it.
class Keyboard
{
public:
  Keyboard()
  {
    const int fd = fileno(stdin);
    if (!isatty(fd)) {
      throw std::runtime_error("keyboard command source requires an interactive TTY");
    }
    if (tcgetattr(fd, &_old_settings) != 0) {
      throw std::runtime_error("tcgetattr(stdin) failed");
    }
    _new_settings = _old_settings;
    _new_settings.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    _terminal_ready = true;
    start_terminal();
    start_reader();
  }

  ~Keyboard()
  {
    stop_reader();
    pause_terminal();
  }

  Keyboard(const Keyboard&) = delete;
  Keyboard& operator=(const Keyboard&) = delete;

  void update()
  {
    const std::string current = key();
    if (current != _last_key) {
      on_pressed = !current.empty();
      on_released = current.empty();
    } else {
      on_pressed = false;
      on_released = false;
    }
    _last_key = current;
  }

  std::string key() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _key;
  }

  // Return the newest key and discard older repeats. This bounds control
  // latency and gives a later Space event priority over queued motion keys.
  bool pop_latest_key(std::string& value)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_events.empty()) return false;
    value = _events.back();
    _events.clear();
    return true;
  }

  bool healthy() const { return _healthy.load(); }

  std::string getString(const std::string& prompt)
  {
    stop_reader();
    pause_terminal();
    std::cout << prompt << std::endl;
    std::string value;
    std::getline(std::cin, value);
    start_terminal();
    start_reader();
    return value;
  }

  bool on_pressed = false;
  bool on_released = false;

private:
  void start_reader()
  {
    _healthy.store(true);
    _running.store(true);
    _read_thread = std::thread([this] {
      while (_running.load()) read_once();
    });
  }

  void stop_reader()
  {
    _running.store(false);
    if (_read_thread.joinable()) _read_thread.join();
  }

  void read_once()
  {
    const int fd = fileno(stdin);
    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(fd, &read_set);
    timeval timeout{0, 80000};
    const int selected = select(fd + 1, &read_set, nullptr, nullptr, &timeout);
    if (selected < 0) {
      if (errno == EINTR) return;
      _healthy.store(false);
      _running.store(false);
      return;
    }
    if (selected == 0) {
      std::lock_guard<std::mutex> lock(_mutex);
      _key.clear();
      return;
    }

    char character = '\0';
    const ssize_t count = read(fd, &character, 1);
    if (count <= 0) {
      _healthy.store(false);
      _running.store(false);
      return;
    }

    // WASD/Space are single-byte ASCII. Escape sequences are deliberately not
    // interpreted so an incomplete sequence can never block the reader.
    std::string parsed;
    if (character != '\033') parsed.assign(1, character);
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _key = parsed;
      if (!parsed.empty()) {
        if (_events.size() >= 64) _events.pop_front();
        _events.push_back(parsed);
      }
    }
  }

  void pause_terminal()
  {
    if (_terminal_ready) tcsetattr(fileno(stdin), TCSANOW, &_old_settings);
  }

  void start_terminal()
  {
    if (_terminal_ready) tcsetattr(fileno(stdin), TCSANOW, &_new_settings);
  }

  mutable std::mutex _mutex;
  std::deque<std::string> _events;
  std::string _key;
  std::string _last_key;
  std::atomic<bool> _running{false};
  std::atomic<bool> _healthy{false};
  std::thread _read_thread;
  termios _old_settings{};
  termios _new_settings{};
  bool _terminal_ready = false;
};
