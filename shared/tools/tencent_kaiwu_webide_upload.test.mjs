import assert from "node:assert/strict";
import test from "node:test";

import {
  normalizeRemoteTarget,
  parseKaiwuSyncBootstrapOutput,
  runKaiwuInteractiveTerminalSmoke,
  startKaiwuSyncService,
  uploadKaiwuWebIdeFile,
} from "./tencent_kaiwu_webide_upload.mjs";

const projectRoot = "/data/projects/legged_robot_competition_26/";
const root = `${projectRoot}agent_ppo/test_artifacts/`;

test("accepts a target under the configured project root", () => {
  assert.equal(
    normalizeRemoteTarget(`${root}agent_ppo/test.bin`, root),
    `${root}agent_ppo/test.bin`,
  );
});

test("rejects traversal outside the configured project root", () => {
  assert.throws(
    () => normalizeRemoteTarget(`${root}../outside.bin`, root),
    /must stay under/,
  );
});

test("rejects the project root as a file target", () => {
  assert.throws(() => normalizeRemoteTarget(root, root), /must stay under/);
});

test("rejects NUL bytes", () => {
  assert.throws(
    () => normalizeRemoteTarget(`${root}bad\0name.bin`, root),
    /without NUL bytes/,
  );
});

test("defaults to the narrow test-artifacts upload root", async () => {
  const { KAIWU_WEBIDE_UPLOAD_DEFAULTS } = await import(
    "./tencent_kaiwu_webide_upload.mjs"
  );
  assert.equal(KAIWU_WEBIDE_UPLOAD_DEFAULTS.remoteRoot, root);
  assert.equal(KAIWU_WEBIDE_UPLOAD_DEFAULTS.chunkSize, 256 * 1024);
  assert.equal(KAIWU_WEBIDE_UPLOAD_DEFAULTS.maxUploadBytes, 4 * 1024 * 1024 * 1024);
  assert.equal(KAIWU_WEBIDE_UPLOAD_DEFAULTS.verificationBackend, "terminal_sha256");
  assert.equal(KAIWU_WEBIDE_UPLOAD_DEFAULTS.writeConcurrency, 1);
  assert.throws(
    () => normalizeRemoteTarget(`${projectRoot}server/agent.py`),
    /must stay under/,
  );
});

test("rejects credentials even when a caller widens the root", () => {
  for (const target of [
    `${projectRoot}conf/.env`,
    `${projectRoot}secrets.json`,
    `${projectRoot}.git/config`,
    `${projectRoot}keys/deploy.pem`,
  ]) {
    assert.throws(
      () => normalizeRemoteTarget(target, projectRoot),
      /protected credential or repository metadata/,
    );
  }
});

test("rejects undeclared verification backends before browser access", async () => {
  await assert.rejects(
    uploadKaiwuWebIdeFile({
      tab: { capabilities: { get() {} }, url() {} },
      localPath: "/unused",
      remotePath: `${root}unused.bin`,
      verificationBackend: "arbitrary_command",
    }),
    /verificationBackend must be terminal_sha256 or browser_readback/,
  );
});

test("interactive terminal smoke validates its timeout before browser access", async () => {
  await assert.rejects(
    runKaiwuInteractiveTerminalSmoke({
      tab: { capabilities: { get() {} }, url() {} },
      timeoutMs: 0,
    }),
    /timeoutMs must be a positive safe integer/,
  );
});

test("sync service bootstrap validates its timeout before browser access", async () => {
  await assert.rejects(
    startKaiwuSyncService({
      tab: { capabilities: { get() {} }, url() {} },
      timeoutMs: 0,
    }),
    /timeoutMs must be a positive safe integer/,
  );
});

test("sync bootstrap parser ignores markers echoed inside the shell command", () => {
  const output = [
    "sh-5.2# if true; then printf 'CODEX_SYNC_STATE=already_running\\n'; fi",
    "CODEX_SYNC_STATE=started",
    "CODEX_SYNC_PID=42",
    "CODEX_SYNC_HEALTH=failed",
  ].join("\r\n");
  assert.deepEqual(parseKaiwuSyncBootstrapOutput(output), {
    state: "started",
    health: "failed",
    pid: 42,
  });
});
