import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { open, stat } from "node:fs/promises";
import path from "node:path";

const DEFAULT_COMMIT = "e2c489dd00f163b1a8d959965b0c30c1a978a080";
const DEFAULT_CHUNK_SIZE = 256 * 1024;
const DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024;
const DEFAULT_REMOTE_ROOT = "/data/projects/legged_robot_competition_26/agent_ppo/test_artifacts/";

function assertUnprotectedRemoteTarget(remotePath) {
  const segments = remotePath.split("/").filter(Boolean);
  const basename = segments.at(-1)?.toLowerCase() || "";
  if (
    segments.includes(".git")
    || /^\.env(?:\..*)?$/.test(basename)
    || /^(?:credentials?|secrets?)(?:\..*)?$/.test(basename)
    || /\.(?:key|pem|p12|pfx)$/.test(basename)
  ) {
    throw new Error("remotePath targets protected credential or repository metadata");
  }
}

function assertPositiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError(`${name} must be a positive safe integer`);
  }
}

function shellArgument(value) {
  return `'${String(value).replaceAll("'", `'\\''`)}'`;
}

export function parseKaiwuSyncBootstrapOutput(output) {
  const resultLines = String(output)
    .replace(/\x1B\[[0-?]*[ -/]*[@-~]/g, "")
    .split(/\r?\n/)
    .map((line) => line.trim());
  const states = resultLines
    .map((line) => /^CODEX_SYNC_STATE=(already_running|started|script_missing)$/.exec(line)?.[1])
    .filter(Boolean);
  const healthStates = resultLines
    .map((line) => /^CODEX_SYNC_HEALTH=(healthy|failed)$/.exec(line)?.[1])
    .filter(Boolean);
  const pidLine = resultLines.find((line) => /^CODEX_SYNC_PID=\d+$/.test(line));
  return {
    state: states.at(-1),
    health: healthStates.at(-1),
    pid: pidLine ? Number(pidLine.slice("CODEX_SYNC_PID=".length)) : null,
  };
}

export function normalizeRemoteTarget(remotePath, remoteRoot = DEFAULT_REMOTE_ROOT) {
  if (typeof remotePath !== "string" || remotePath.includes("\0")) {
    throw new TypeError("remotePath must be a non-empty path without NUL bytes");
  }

  const normalizedRoot = `${path.posix.resolve("/", remoteRoot)}/`;
  const normalizedTarget = path.posix.resolve("/", remotePath);
  if (!normalizedTarget.startsWith(normalizedRoot) || normalizedTarget === normalizedRoot.slice(0, -1)) {
    throw new Error(`remotePath must stay under ${normalizedRoot}`);
  }
  assertUnprotectedRemoteTarget(normalizedTarget);
  return normalizedTarget;
}

async function sha256File(filePath) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(filePath)) {
    hash.update(chunk);
  }
  return hash.digest("hex");
}

async function evaluate(cdp, expression, timeoutMs = 30_000) {
  const response = await cdp.send(
    "Runtime.evaluate",
    { expression, awaitPromise: true, returnByValue: true },
    { timeoutMs },
  );
  if (response.exceptionDetails) {
    const description = response.result?.description || response.exceptionDetails.text || "unknown browser error";
    throw new Error(description);
  }
  return response.result?.value;
}

async function remoteSha256(cdp, session, remotePath, timeoutMs = 180_000) {
  await evaluate(cdp, `${session}.startSha256(${JSON.stringify(remotePath)})`);
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const status = await evaluate(cdp, `${session}.sha256Status()`);
    if (status?.state === "done") return status.sha256;
    if (status?.state === "error") throw new Error(status.message || "remote SHA256 failed");
    if (Date.now() >= deadline) throw new Error("remote SHA256 timed out");
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

async function remoteTerminalSha256(
  cdp,
  session,
  remotePath,
  resultPath,
  timeoutMs = 180_000,
) {
  await evaluate(
    cdp,
    `${session}.startTerminalSha256(${JSON.stringify(remotePath)},${JSON.stringify(resultPath)})`,
    40_000,
  );
  const deadline = Date.now() + timeoutMs;
  let lastError;
  for (;;) {
    const status = await evaluate(
      cdp,
      `${session}.readTextIfExists(${JSON.stringify(resultPath)})`,
    );
    if (status?.found) {
      const match = /^([a-f0-9]{64})(?:\s|$)/i.exec(status.text || "");
      if (match) return match[1].toLowerCase();
      if (/^ERROR\s+/.test(status.text || "")) {
        throw new Error(`container sha256sum failed: ${status.text.trim()}`);
      }
      throw new Error("container sha256sum returned an invalid result");
    }
    lastError = status?.message || lastError;
    if (Date.now() >= deadline) {
      throw new Error(`container sha256sum timed out${lastError ? `: ${lastError}` : ""}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

function buildBootstrapExpression({ authority, commit, ideId, sessionKey, token }) {
  const endpoint = `wss://${authority}/p5/ide/${ideId}/stable-${commit}`;
  return String.raw`
(async () => {
  const endpoint = ${JSON.stringify(endpoint)};
  const token = ${JSON.stringify(token)};
  const commit = ${JSON.stringify(commit)};
  const authority = ${JSON.stringify(authority)};
  const sessionKey = ${JSON.stringify(sessionKey)};
  const te = new TextEncoder();
  const td = new TextDecoder();

  const varuint = (value) => {
    const bytes = [];
    do {
      let byte = value & 0x7f;
      value = Math.floor(value / 128);
      if (value) byte |= 0x80;
      bytes.push(byte);
    } while (value);
    return bytes;
  };
  const concat = (...parts) => {
    const size = parts.reduce((total, part) => total + part.length, 0);
    const result = new Uint8Array(size);
    let offset = 0;
    for (const part of parts) {
      result.set(part, offset);
      offset += part.length;
    }
    return result;
  };
  const encode = (value) => {
    if (value === undefined) return Uint8Array.of(0);
    if (value === null || typeof value === "boolean") {
      const bytes = te.encode(JSON.stringify(value));
      return concat(Uint8Array.of(5), Uint8Array.from(varuint(bytes.length)), bytes);
    }
    if (typeof value === "string") {
      const bytes = te.encode(value);
      return concat(Uint8Array.of(1), Uint8Array.from(varuint(bytes.length)), bytes);
    }
    if (value instanceof Uint8Array) {
      return concat(Uint8Array.of(3), Uint8Array.from(varuint(value.length)), value);
    }
    if (Array.isArray(value)) {
      return concat(Uint8Array.of(4), Uint8Array.from(varuint(value.length)), ...value.map(encode));
    }
    if (value && typeof value === "object") {
      const bytes = te.encode(JSON.stringify(value));
      return concat(Uint8Array.of(5), Uint8Array.from(varuint(bytes.length)), bytes);
    }
    if (Number.isSafeInteger(value) && value >= 0) {
      return concat(Uint8Array.of(6), Uint8Array.from(varuint(value)));
    }
    throw new Error("unsupported protocol value");
  };
  const readVaruint = (bytes, start) => {
    let value = 0;
    let multiplier = 1;
    let position = start;
    for (;;) {
      const byte = bytes[position++];
      value += (byte & 0x7f) * multiplier;
      if (!(byte & 0x80)) return [value, position];
      multiplier *= 128;
    }
  };
  const decode = (bytes, start = 0) => {
    const tag = bytes[start++];
    if (tag === 0) return [undefined, start];
    let length;
    if (tag === 1 || tag === 2 || tag === 3) {
      [length, start] = readVaruint(bytes, start);
      const end = start + length;
      if (tag === 1) return [td.decode(bytes.slice(start, end)), end];
      return [bytes.slice(start, end), end];
    }
    if (tag === 4) {
      [length, start] = readVaruint(bytes, start);
      const result = [];
      for (let index = 0; index < length; index++) {
        let value;
        [value, start] = decode(bytes, start);
        result.push(value);
      }
      return [result, start];
    }
    if (tag === 5) {
      [length, start] = readVaruint(bytes, start);
      const end = start + length;
      return [JSON.parse(td.decode(bytes.slice(start, end))), end];
    }
    if (tag === 6) {
      let value;
      [value, start] = readVaruint(bytes, start);
      return [value, start];
    }
    throw new Error("unsupported protocol tag " + tag);
  };
  const packet = (type, id, ack, body) => {
    const result = new Uint8Array(13 + body.length);
    const view = new DataView(result.buffer);
    result[0] = type;
    view.setUint32(1, id);
    view.setUint32(5, ack);
    view.setUint32(9, body.length);
    result.set(body, 13);
    return result;
  };
  const special = (value) => packet(2, 0, 0, te.encode(JSON.stringify(value)));
  const uri = (remotePath) => ({
    $mid: 1,
    external: "vscode-remote://" + authority + remotePath,
    path: remotePath,
    scheme: "vscode-remote",
    authority,
  });

  let socket;
  let messageId = 0;
  let ack = 0;
  let requestId = 0;
  let receiveBuffer = new Uint8Array(0);
  let fileDescriptor = null;
  let hashJob = { state: "idle" };
  let terminalJob = { state: "idle", output: "" };
  const pending = new Map();
  const listeners = new Map();
  const rejectPending = (error) => {
    for (const entry of pending.values()) entry.reject(error);
    pending.clear();
  };
  const callChannel = (channel, method, args) => new Promise((resolve, reject) => {
    const id = ++requestId;
    const timer = setTimeout(() => {
      if (pending.delete(id)) reject(new Error(channel + " timeout: " + method));
    }, 30_000);
    pending.set(id, {
      resolve: (value) => { clearTimeout(timer); resolve(value); },
      reject: (error) => { clearTimeout(timer); reject(error); },
    });
    const body = concat(encode([100, id, channel, method]), encode(args));
    socket.send(packet(1, ++messageId, ack, body));
  });
  const call = (method, args) => callChannel("remoteFilesystem", method, args);
  const terminalCall = (method, args) => callChannel("remoteterminal", method, args);
  const listenChannel = (channel, eventName, args, listener) => {
    const id = ++requestId;
    listeners.set(id, listener);
    const body = concat(encode([102, id, channel, eventName]), encode(args));
    socket.send(packet(1, ++messageId, ack, body));
    return id;
  };
  const shellQuote = (value) => "'" + value.replaceAll("'", "'\\''") + "'";

  return await new Promise((resolve, reject) => {
    const connectionTimer = setTimeout(() => reject(new Error("Remote Agent connection timeout")), 30_000);
    socket = new WebSocket(endpoint + "?reconnectionToken=" + token + "&reconnection=false&skipWebSocketFrames=false");
    socket.binaryType = "arraybuffer";
    socket.onerror = () => {
      const error = new Error("Remote Agent WebSocket error");
      rejectPending(error);
      reject(error);
    };
    socket.onclose = () => rejectPending(new Error("Remote Agent WebSocket closed"));
    socket.onopen = () => socket.send(special({
      type: "auth",
      auth: "00000000000000000000",
      data: token,
    }));
    socket.onmessage = async (event) => {
      try {
        receiveBuffer = concat(receiveBuffer, new Uint8Array(event.data));
        let position = 0;
        while (position + 13 <= receiveBuffer.length) {
          const type = receiveBuffer[position];
          const view = new DataView(receiveBuffer.buffer, receiveBuffer.byteOffset + position);
          const id = view.getUint32(1);
          const length = view.getUint32(9);
          if (position + 13 + length > receiveBuffer.length) break;
          const body = receiveBuffer.slice(position + 13, position + 13 + length);
          position += 13 + length;

          if (type === 2) {
            const value = JSON.parse(td.decode(body));
            if (value.type === "sign") {
              socket.send(special({
                type: "connectionType",
                commit,
                signedData: value.data,
                desiredConnectionType: 1,
              }));
            } else if (value.type === "ok") {
              socket.send(concat(
                packet(1, ++messageId, ack, encode({ remoteAuthority: authority, clientId: "renderer" })),
                packet(1, ++messageId, ack, concat(encode([200]), encode(undefined))),
              ));
            }
            continue;
          }

          if (type !== 1) continue;
          ack = Math.max(ack, id);
          let header;
          let valuePosition;
          [header, valuePosition] = decode(body);
          if (Array.isArray(header) && header[0] === 200 && !window[sessionKey]) {
            listenChannel("remoteterminal", "$onProcessDataEvent", undefined, (payload) => {
              if (payload?.id !== terminalJob.processId) return;
              const event = payload.event;
              const data = typeof event === "string" ? event : event?.data;
              if (typeof data !== "string") return;
              terminalJob.output += data;
              terminalCall("$acknowledgeDataEvent", [payload.id, data.length]).catch(() => {});
            });
            listenChannel("remoteterminal", "$onProcessReadyEvent", undefined, (payload) => {
              if (payload?.id !== terminalJob.processId) return;
              terminalJob.state = "ready";
              terminalJob.pid = payload.event?.pid;
              terminalJob.cwd = payload.event?.cwd;
            });
            listenChannel("remoteterminal", "$onProcessExitEvent", undefined, (payload) => {
              if (payload?.id !== terminalJob.processId) return;
              terminalJob.state = "exited";
              terminalJob.exitCode = payload.event;
            });
            const uploader = {
              async openWrite(remotePath) {
                fileDescriptor = await call("open", [uri(remotePath), { create: true }]);
              },
              async writeBase64(writePosition, base64) {
                const raw = atob(base64);
                const bytes = new Uint8Array(raw.length);
                for (let index = 0; index < raw.length; index++) bytes[index] = raw.charCodeAt(index);
                return await call("write", [fileDescriptor, writePosition, bytes, 0, bytes.length]);
              },
              async closeWrite() {
                if (fileDescriptor !== null) {
                  await call("close", [fileDescriptor]);
                  fileDescriptor = null;
                }
              },
              async stat(remotePath) {
                return await call("stat", [uri(remotePath)]);
              },
              startSha256(remotePath) {
                if (hashJob.state === "running") throw new Error("remote SHA256 already running");
                hashJob = { state: "running" };
                (async () => {
                  try {
                    const bytes = await call("readFile", [uri(remotePath)]);
                    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
                    hashJob = {
                      state: "done",
                      sha256: Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join(""),
                    };
                  } catch (error) {
                    hashJob = { state: "error", message: error?.message || String(error) };
                  }
                })();
                return true;
              },
              sha256Status() {
                return hashJob;
              },
              async startTerminalSha256(remotePath, resultPath) {
                const resultTmpPath = resultPath + ".tmp";
                const command = [
                  "/usr/bin/sha256sum -- " + shellQuote(remotePath),
                  " > " + shellQuote(resultTmpPath),
                  " && /bin/mv -- " + shellQuote(resultTmpPath) + " " + shellQuote(resultPath),
                  " || { code=$?; /usr/bin/printf 'ERROR %s\\n' \"$code\" > " + shellQuote(resultPath) + "; }",
                ].join("");
                const created = await terminalCall("$createProcess", {
                  configuration: {
                    "terminal.integrated.cwd": "",
                    "terminal.integrated.detectLocale": "off",
                    "terminal.integrated.env.linux": {},
                    "terminal.integrated.env.osx": {},
                    "terminal.integrated.env.windows": {},
                  },
                  resolvedVariables: {},
                  envVariableCollections: [],
                  shellLaunchConfig: {
                    executable: "/bin/sh",
                    args: ["-c", command],
                    cwd: remotePath.slice(0, remotePath.lastIndexOf("/")) || "/",
                    hideFromUser: true,
                  },
                  workspaceId: "",
                  workspaceName: "",
                  workspaceFolders: [],
                  activeWorkspaceFolder: null,
                  shouldPersistTerminal: false,
                  options: {
                    shellIntegration: { enabled: false, suggestEnabled: false, nonce: "" },
                    windowsEnableConpty: false,
                    windowsUseConptyDll: false,
                  },
                  cols: 80,
                  rows: 24,
                  unicodeVersion: "11",
                });
                const processId = created?.persistentTerminalId;
                if (!Number.isSafeInteger(processId)) {
                  throw new Error("remoteterminal did not return a process id");
                }
                const launchError = await terminalCall("$start", [processId]);
                if (launchError) {
                  throw new Error(launchError.message || "remoteterminal process launch failed");
                }
                return { processId };
              },
              async readTextIfExists(remotePath) {
                try {
                  const bytes = await call("readFile", [uri(remotePath)]);
                  return { found: true, text: td.decode(bytes) };
                } catch (error) {
                  return { found: false, message: error?.message || String(error) };
                }
              },
              async startInteractiveTerminal(cwd) {
                terminalJob = { state: "starting", output: "" };
                const created = await terminalCall("$createProcess", {
                  configuration: {
                    "terminal.integrated.cwd": "",
                    "terminal.integrated.detectLocale": "off",
                    "terminal.integrated.env.linux": {},
                    "terminal.integrated.env.osx": {},
                    "terminal.integrated.env.windows": {},
                  },
                  resolvedVariables: {},
                  envVariableCollections: [],
                  shellLaunchConfig: {
                    executable: "/bin/sh",
                    args: ["-i"],
                    cwd,
                    hideFromUser: true,
                  },
                  workspaceId: "",
                  workspaceName: "",
                  workspaceFolders: [],
                  activeWorkspaceFolder: null,
                  shouldPersistTerminal: false,
                  options: {
                    shellIntegration: { enabled: false, suggestEnabled: false, nonce: "" },
                    windowsEnableConpty: false,
                    windowsUseConptyDll: false,
                  },
                  cols: 80,
                  rows: 24,
                  unicodeVersion: "11",
                });
                const processId = created?.persistentTerminalId;
                if (!Number.isSafeInteger(processId)) {
                  throw new Error("remoteterminal did not return a process id");
                }
                terminalJob.processId = processId;
                const launchError = await terminalCall("$start", [processId]);
                if (launchError) {
                  throw new Error(launchError.message || "interactive terminal launch failed");
                }
                return { processId };
              },
              async terminalInput(data) {
                if (!Number.isSafeInteger(terminalJob.processId)) {
                  throw new Error("interactive terminal is not running");
                }
                return await terminalCall("$input", [terminalJob.processId, data]);
              },
              terminalStatus() {
                return terminalJob;
              },
              async stopInteractiveTerminal() {
                if (!Number.isSafeInteger(terminalJob.processId) || terminalJob.state === "exited") {
                  return false;
                }
                await terminalCall("$shutdown", [terminalJob.processId, true]);
                return true;
              },
              async rename(source, target) {
                return await call("rename", [uri(source), uri(target), { overwrite: true }]);
              },
              async delete(remotePath) {
                return await call("delete", [uri(remotePath), {
                  recursive: false,
                  useTrash: false,
                  atomic: false,
                }]);
              },
              shutdown() {
                socket.close();
                delete window[sessionKey];
              },
            };
            window[sessionKey] = uploader;
            clearTimeout(connectionTimer);
            resolve({ ready: true });
          } else if (Array.isArray(header) && header[0] === 204) {
            let value;
            [value] = decode(body, valuePosition);
            listeners.get(header[1])?.(value);
          } else if (
            Array.isArray(header)
            && (header[0] === 201 || header[0] === 202 || header[0] === 203)
          ) {
            let value;
            [value] = decode(body, valuePosition);
            const entry = pending.get(header[1]);
            if (entry) {
              pending.delete(header[1]);
              if (header[0] === 201) entry.resolve(value);
              else entry.reject(new Error(value?.message || "Remote Agent channel error"));
            }
          }
        }
        receiveBuffer = receiveBuffer.slice(position);
      } catch (error) {
        rejectPending(error);
        reject(error);
      }
    };
  });
})()`;
}

export async function uploadKaiwuWebIdeFile({
  tab,
  localPath,
  remotePath,
  remoteRoot = DEFAULT_REMOTE_ROOT,
  ideId = "18005",
  authority = "tencentarena.com",
  commit = DEFAULT_COMMIT,
  chunkSize = DEFAULT_CHUNK_SIZE,
  maxUploadBytes = DEFAULT_MAX_UPLOAD_BYTES,
  deleteAfterVerify = false,
  verificationMode = "staging_sha256_final_stat",
  verificationBackend = "terminal_sha256",
}) {
  if (!tab?.capabilities?.get || typeof tab.url !== "function") {
    throw new TypeError("tab must be a controlled Chrome tab from the browser client");
  }
  assertPositiveInteger(chunkSize, "chunkSize");
  assertPositiveInteger(maxUploadBytes, "maxUploadBytes");
  if (chunkSize > DEFAULT_CHUNK_SIZE) {
    throw new Error(`chunkSize must not exceed the verified ${DEFAULT_CHUNK_SIZE} byte limit`);
  }
  if (!["double_sha256", "staging_sha256_final_stat"].includes(verificationMode)) {
    throw new Error("verificationMode must be double_sha256 or staging_sha256_final_stat");
  }
  if (!["terminal_sha256", "browser_readback"].includes(verificationBackend)) {
    throw new Error("verificationBackend must be terminal_sha256 or browser_readback");
  }

  const target = normalizeRemoteTarget(remotePath, remoteRoot);
  const staging = `${target}.uploading`;
  const localInfo = await stat(localPath);
  if (!localInfo.isFile()) throw new Error("localPath must point to a regular file");
  if (localInfo.size > maxUploadBytes) {
    throw new Error(
      `file is ${localInfo.size} bytes; upload limit is ${maxUploadBytes} bytes`,
    );
  }

  const currentUrl = await tab.url();
  if (!currentUrl?.startsWith(`https://${authority}/`)) {
    throw new Error(`Chrome tab must be on the authenticated ${authority} IDE page`);
  }

  const localSha256 = await sha256File(localPath);
  const cdp = await tab.capabilities.get("cdp");
  const sessionKey = `__codexKaiwuUpload_${randomUUID().replaceAll("-", "")}`;
  const token = randomUUID();
  const bootstrap = buildBootstrapExpression({ authority, commit, ideId, sessionKey, token });
  await evaluate(cdp, bootstrap, 40_000);

  const session = `window[${JSON.stringify(sessionKey)}]`;
  const startedAt = Date.now();
  let uploadedBytes = 0;
  let chunkCount = 0;
  let renamed = false;
  let uploadCompletedAt;
  let stagingVerifiedAt;
  const pendingHashArtifacts = new Set();
  const verifyRemote = async (remotePath) => {
    if (verificationBackend === "browser_readback") {
      return await remoteSha256(cdp, session, remotePath);
    }
    const resultPath = `${remotePath}.sha256-${randomUUID()}`;
    const artifactPaths = [resultPath, `${resultPath}.tmp`];
    for (const artifactPath of artifactPaths) pendingHashArtifacts.add(artifactPath);
    try {
      return await remoteTerminalSha256(cdp, session, remotePath, resultPath);
    } finally {
      for (const artifactPath of artifactPaths) {
        try {
          await evaluate(cdp, `${session}.delete(${JSON.stringify(artifactPath)})`);
          pendingHashArtifacts.delete(artifactPath);
        } catch {}
      }
    }
  };
  const file = await open(localPath, "r");
  try {
    await evaluate(cdp, `${session}.openWrite(${JSON.stringify(staging)})`);
    const buffer = Buffer.allocUnsafe(chunkSize);
    for (;;) {
      const { bytesRead } = await file.read(buffer, 0, buffer.length, uploadedBytes);
      if (!bytesRead) break;
      const base64 = buffer.subarray(0, bytesRead).toString("base64");
      const written = await evaluate(
        cdp,
        `${session}.writeBase64(${uploadedBytes},${JSON.stringify(base64)})`,
        40_000,
      );
      if (written !== bytesRead) {
        throw new Error(`short remote write at ${uploadedBytes}: ${written}/${bytesRead}`);
      }
      uploadedBytes += bytesRead;
      chunkCount += 1;
    }
    await evaluate(cdp, `${session}.closeWrite()`);
    uploadCompletedAt = Date.now();

    const stagingInfo = {
      stat: await evaluate(cdp, `${session}.stat(${JSON.stringify(staging)})`),
      sha256: await verifyRemote(staging),
    };
    if (stagingInfo.stat?.size !== localInfo.size || stagingInfo.sha256 !== localSha256) {
      throw new Error("remote staging size or SHA256 mismatch");
    }
    stagingVerifiedAt = Date.now();

    await evaluate(
      cdp,
      `${session}.rename(${JSON.stringify(staging)},${JSON.stringify(target)})`,
    );
    renamed = true;
    const finalInfo = {
      stat: await evaluate(cdp, `${session}.stat(${JSON.stringify(target)})`),
    };
    if (verificationMode === "double_sha256") {
      finalInfo.sha256 = await verifyRemote(target);
    }
    if (finalInfo.stat?.size !== localInfo.size) {
      throw new Error("remote final size mismatch");
    }
    if (verificationMode === "double_sha256" && finalInfo.sha256 !== localSha256) {
      throw new Error("remote final SHA256 mismatch");
    }

    if (deleteAfterVerify) {
      await evaluate(cdp, `${session}.delete(${JSON.stringify(target)})`);
    }

    const completedAt = Date.now();
    const elapsedMs = completedAt - startedAt;
    return {
      localPath,
      remotePath: target,
      bytes: localInfo.size,
      sha256: localSha256,
      chunkSize,
      chunkCount,
      writeConcurrency: 1,
      elapsedMs,
      uploadElapsedMs: uploadCompletedAt - startedAt,
      stagingVerifyElapsedMs: stagingVerifiedAt - uploadCompletedAt,
      finalVerifyElapsedMs: completedAt - stagingVerifiedAt,
      mibPerSecond: elapsedMs ? (localInfo.size / 1024 / 1024) / (elapsedMs / 1000) : null,
      uploadMibPerSecond: uploadCompletedAt > startedAt
        ? (localInfo.size / 1024 / 1024) / ((uploadCompletedAt - startedAt) / 1000)
        : null,
      protocolCommit: commit,
      deletedAfterVerify: deleteAfterVerify,
      verificationMode,
      verificationBackend,
      finalSha256Verified: verificationMode === "double_sha256",
    };
  } catch (error) {
    if (!renamed) {
      try {
        await evaluate(cdp, `${session}.closeWrite()`);
      } catch {}
      try {
        await evaluate(cdp, `${session}.delete(${JSON.stringify(staging)})`);
      } catch {}
    }
    throw error;
  } finally {
    await file.close();
    for (const cleanupPath of pendingHashArtifacts) {
      try {
        await evaluate(cdp, `${session}.delete(${JSON.stringify(cleanupPath)})`);
      } catch {}
    }
    try {
      await evaluate(cdp, `${session}.shutdown()`);
    } catch {}
  }
}

export async function runKaiwuInteractiveTerminalSmoke({
  tab,
  ideId = "18005",
  authority = "tencentarena.com",
  commit = DEFAULT_COMMIT,
  timeoutMs = 30_000,
}) {
  if (!tab?.capabilities?.get || typeof tab.url !== "function") {
    throw new TypeError("tab must be a controlled Chrome tab from the browser client");
  }
  assertPositiveInteger(timeoutMs, "timeoutMs");
  const currentUrl = await tab.url();
  if (!currentUrl?.startsWith(`https://${authority}/`)) {
    throw new Error(`Chrome tab must be on the authenticated ${authority} IDE page`);
  }

  const cdp = await tab.capabilities.get("cdp");
  const sessionKey = `__codexKaiwuTerminal_${randomUUID().replaceAll("-", "")}`;
  const token = randomUUID();
  await evaluate(
    cdp,
    buildBootstrapExpression({ authority, commit, ideId, sessionKey, token }),
    40_000,
  );
  const session = `window[${JSON.stringify(sessionKey)}]`;
  const marker = `CODEX_PTY_${randomUUID().replaceAll("-", "")}`;
  const commands = [
    `printf '%s\\n' ${marker}\r`,
    "pwd\r",
    "python3 --version\r",
    "exit\r",
  ];
  try {
    await evaluate(
      cdp,
      `${session}.startInteractiveTerminal(${JSON.stringify("/data/projects/legged_robot_competition_26")})`,
      40_000,
    );
    for (const command of commands) {
      await evaluate(cdp, `${session}.terminalInput(${JSON.stringify(command)})`);
    }

    const deadline = Date.now() + timeoutMs;
    let status;
    do {
      status = await evaluate(cdp, `${session}.terminalStatus()`);
      if (status?.state === "exited") break;
      if (Date.now() >= deadline) throw new Error("interactive terminal smoke timed out");
      await new Promise((resolve) => setTimeout(resolve, 100));
    } while (true);

    const output = status.output || "";
    const projectRoot = "/data/projects/legged_robot_competition_26";
    return {
      processId: status.processId,
      exitCode: status.exitCode,
      markerSeen: output.includes(marker),
      cwdSeen: output.includes(projectRoot),
      pythonSeen: /Python\s+\d+\.\d+\.\d+/.test(output),
      output,
    };
  } finally {
    try {
      await evaluate(cdp, `${session}.stopInteractiveTerminal()`);
    } catch {}
    try {
      await evaluate(cdp, `${session}.shutdown()`);
    } catch {}
  }
}

export async function startKaiwuSyncService({
  tab,
  ideId = "18005",
  authority = "tencentarena.com",
  commit = DEFAULT_COMMIT,
  timeoutMs = 30_000,
}) {
  if (!tab?.capabilities?.get || typeof tab.url !== "function") {
    throw new TypeError("tab must be a controlled Chrome tab from the browser client");
  }
  assertPositiveInteger(timeoutMs, "timeoutMs");
  const currentUrl = await tab.url();
  if (!currentUrl?.startsWith(`https://${authority}/`)) {
    throw new Error(`Chrome tab must be on the authenticated ${authority} IDE page`);
  }

  const cdp = await tab.capabilities.get("cdp");
  const sessionKey = `__codexKaiwuSyncStart_${randomUUID().replaceAll("-", "")}`;
  const token = randomUUID();
  await evaluate(
    cdp,
    buildBootstrapExpression({ authority, commit, ideId, sessionKey, token }),
    40_000,
  );
  const session = `window[${JSON.stringify(sessionKey)}]`;
  const projectRoot = "/data/projects/legged_robot_competition_26";
  const scriptPath = `${projectRoot}/conf/start_tongbu.sh`;
  const logPath = "/tmp/codex-tongbu-18005.log";
  const healthUrl = "http://127.0.0.1:8765/health";
  const command = [
    `script=${shellArgument(scriptPath)}; `,
    `log=${shellArgument(logPath)}; `,
    `health=${shellArgument(healthUrl)}; `,
    "if curl -sS --max-time 2 \"$health\" >/dev/null 2>&1; then ",
    "printf 'CODEX_SYNC_STATE=already_running\\n'; ",
    "elif [ ! -f \"$script\" ]; then ",
    "printf 'CODEX_SYNC_STATE=script_missing\\n'; ",
    "else nohup /bin/sh \"$script\" >\"$log\" 2>&1 </dev/null & ",
    "printf 'CODEX_SYNC_STATE=started\\nCODEX_SYNC_PID=%s\\n' \"$!\"; fi; ",
    "i=0; while [ \"$i\" -lt 40 ]; do ",
    "if curl -sS --max-time 2 \"$health\" >/dev/null 2>&1; then break; fi; ",
    "i=$((i + 1)); sleep 0.25; done; ",
    "if curl -sS --max-time 2 \"$health\" >/dev/null 2>&1; then ",
    "printf 'CODEX_SYNC_HEALTH=healthy\\n'; ",
    "else printf 'CODEX_SYNC_HEALTH=failed\\n'; fi",
  ].join("");

  try {
    await evaluate(
      cdp,
      `${session}.startInteractiveTerminal(${JSON.stringify(projectRoot)})`,
      40_000,
    );
    await evaluate(cdp, `${session}.terminalInput(${JSON.stringify(`${command}\r`)})`);
    await evaluate(cdp, `${session}.terminalInput(${JSON.stringify("exit\r")})`);

    const deadline = Date.now() + timeoutMs;
    let status;
    do {
      status = await evaluate(cdp, `${session}.terminalStatus()`);
      if (status?.state === "exited") break;
      if (Date.now() >= deadline) throw new Error("sync service bootstrap timed out");
      await new Promise((resolve) => setTimeout(resolve, 100));
    } while (true);

    const output = status.output || "";
    const parsed = parseKaiwuSyncBootstrapOutput(output);
    const result = {
      alreadyRunning: parsed.state === "already_running",
      started: parsed.state === "started",
      healthy: parsed.health === "healthy",
      scriptMissing: parsed.state === "script_missing",
      pid: parsed.pid,
      exitCode: status.exitCode,
      logPath,
      output,
    };
    if (!result.healthy) {
      throw new Error(
        result.scriptMissing
          ? `sync service script is missing: ${scriptPath}`
          : `sync service did not become healthy; inspect ${logPath}`,
      );
    }
    return result;
  } finally {
    try {
      await evaluate(cdp, `${session}.stopInteractiveTerminal()`);
    } catch {}
    try {
      await evaluate(cdp, `${session}.shutdown()`);
    } catch {}
  }
}

export const KAIWU_WEBIDE_UPLOAD_DEFAULTS = Object.freeze({
  authority: "tencentarena.com",
  chunkSize: DEFAULT_CHUNK_SIZE,
  commit: DEFAULT_COMMIT,
  ideId: "18005",
  maxUploadBytes: DEFAULT_MAX_UPLOAD_BYTES,
  remoteRoot: DEFAULT_REMOTE_ROOT,
  verificationBackend: "terminal_sha256",
  writeConcurrency: 1,
});
