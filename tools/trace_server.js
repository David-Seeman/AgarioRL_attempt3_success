// Ground-truth tracer: runs the UNMODIFIED imsolo server code (reference_server/)
// headlessly, feeds it a tick-by-tick input tape for two players, and dumps every
// player cell after every tick.  Used to verify splitrun/sim.py cell-for-cell.
//
//   node tools/trace_server.js tape.json > trace.json
//
// tape = { seed, cells: [{owner, x, y, size, age, boost?: {dx, dy, d}}],
//          inputs: [ [[mx, my, splits], [mx, my, splits]], ... one entry per tick ] }
// out  = { ticks: [ [[id, owner, x, y, size, boostD], ...playerCells order], ... ] }

const path = require("path");
const fs = require("fs");
const Module = require("module");

// The websocket library is never used headlessly; stub it so no npm install is needed.
const origLoad = Module._load;
Module._load = function (req, ...rest) {
    if (req === "uws") return { Server: class { } };
    return origLoad.call(this, req, ...rest);
};

const SRC = path.join(__dirname, "..", "reference_server", "Megasplit-Beta", "src");
const ServerHandle = require(path.join(SRC, "ServerHandle"));
const Router = require(path.join(SRC, "sockets", "Router"));
const PlayerCell = require(path.join(SRC, "cells", "PlayerCell"));
const FFA = require(path.join(SRC, "gamemodes", "FFA"));

const tape = JSON.parse(fs.readFileSync(process.argv[2], "utf-8"));
const envSettings = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "env_settings.json"), "utf-8"));

// Deterministic Math.random (mulberry32) so autosplit angles can be reproduced.
let rngState = (tape.seed >>> 0) || 1;
Math.random = function () {
    rngState = (rngState + 0x6D2B79F5) >>> 0;
    let t = rngState;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

class TapeRouter extends Router {
    static get type() { return "tape"; }
    static get isExternal() { return false; }
    static get separateInTeams() { return false; }
    get shouldClose() { return false; }
    update() { }
}

const handle = new ServerHandle(Object.assign({}, envSettings, { worldMinCount: 5 }));
handle.gamemodes.register(FFA);
handle.gamemodes.setGamemode("FFA");
handle.tick = 0;
handle.startTime = new Date();
const world = handle.createWorld();

const routers = [0, 1].map(() => {
    const r = new TapeRouter(handle.listener);
    r.createPlayer();
    world.addPlayer(r.player);
    r.player.cellName = "";
    return r;
});

for (const c of tape.cells) {
    const cell = new PlayerCell(routers[c.owner].player, c.x, c.y, c.size);
    cell.birthTick = handle.tick - (c.age || 0);
    world.addCell(cell);
    if (c.boost && c.boost.d > 0) {
        cell.boost.dx = c.boost.dx; cell.boost.dy = c.boost.dy; cell.boost.d = c.boost.d;
        world.setCellAsBoosting(cell);
    }
}
for (const r of routers) r.player.updateState(0);

const ticks = [];
for (const inp of tape.inputs) {
    for (let p = 0; p < 2; p++) {
        // the wire protocol delivers integer mouse coordinates
        routers[p].mouseX = ~~inp[p][0];
        routers[p].mouseY = ~~inp[p][1];
        routers[p].splitAttempts = inp[p][2];
    }
    handle.tick++;
    world.liveUpdate();
    ticks.push(world.playerCells.map(c => [c.id, c.owner === routers[0].player ? 0 : 1, c.x, c.y, c.size, c.isBoosting ? c.boost.d : 0]));
    if (routers[0].player.ownedCells.length === 0 || routers[1].player.ownedCells.length === 0) break;
}
process.stdout.write(JSON.stringify({ ticks }));
