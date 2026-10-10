// ============================================================================
// 7. Field Canvas 2D Renderer (With X-Drive 45° Wheels & Proportional Elements)
// ============================================================================
class FieldRenderer {
  constructor(canvas, simulator) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.sim = simulator;

    this.showTargetPath = true;
    this.showTrail = true;
    this.showEKF = true;
    this.showCoordinates = true;
    this.showFieldElements = true;

    // Jerry.io Planner State
    this.jerryWaypoints = [];
    this.selectedWpIndex = -1;
    this.draggedWpIndex = -1;
    this.dragMode = 'none'; // 'pos' or 'heading'

    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  resizeCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const size = Math.min(rect.width, rect.height || rect.width);
    this.canvas.width = size * window.devicePixelRatio;
    this.canvas.height = size * window.devicePixelRatio;
    this.scale = this.canvas.width / 140.4;
  }

  toCanvas(xInches, yInches) {
    const cx = this.canvas.width / 2;
    const cy = this.canvas.height / 2;
    return {
      x: cx + (xInches * this.scale),
      y: cy - (yInches * this.scale)
    };
  }

  toField(px, py) {
    const cx = this.canvas.width / 2;
    const cy = this.canvas.height / 2;
    return {
      x: (px - cx) / this.scale,
      y: (cy - py) / this.scale
    };
  }

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this.drawFieldBackground();
    if (this.showFieldElements) this.drawFieldElements();
    if (this.showTargetPath) this.drawPlannedTrajectory();
    this.drawJerryWaypoints();
    if (this.showTrail) this.drawPathTrail();
    if (this.showEKF) this.drawEKFEllipse();
    this.drawRobot();
  }

  drawFieldBackground() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const tileSize = 23.4 * this.scale;

    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        const x = c * tileSize;
        const y = r * tileSize;
        ctx.fillStyle = (r + c) % 2 === 0 ? '#737982' : '#808791';
        ctx.fillRect(x, y, tileSize, tileSize);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, tileSize, tileSize);
      }
    }

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
    ctx.lineWidth = 3;

    // Diagonal quadrant boundaries; central diamond is drawn by OverrideView.
    for(const [a,b] of [[[-60,60],[-11.555,11.555]],[[11.555,-11.555],[60,-60]],[[-60,-60],[-11.555,-11.555]],[[11.555,11.555],[60,60]]]){
      const p=this.toCanvas(...a),q=this.toCanvas(...b);ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();
    }
    ctx.strokeStyle = '#384661';
    ctx.lineWidth = 8;
    ctx.strokeRect(4, 4, w - 8, h - 8);

    ctx.fillStyle = '#f43f5e';
    ctx.fillRect(4, 4, 28, 28);
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(w - 32, h - 32, 28, 28);

    if (this.showCoordinates) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
      ctx.font = `${Math.max(9, 10 * (this.scale / 4.5))}px JetBrains Mono`;
      for (let inX = -48; inX <= 48; inX += 24) {
        for (let inY = -48; inY <= 48; inY += 48) {
          const pt = this.toCanvas(inX, inY);
          ctx.fillText(`(${inX}", ${inY}")`, pt.x + 4, pt.y - 4);
        }
      }
    }
  }

  // Reduced, proportional game elements (User request: smaller elements)
  drawFieldElements() {
    const ctx = this.ctx;
    if(this.sim.override && typeof OverrideView!=="undefined"){OverrideView.draw2D(this);return;}

    // Center Ladder: reduced from 10 to 5.5 inches
    const center = this.toCanvas(0, 0);
    const ladderRadius = 5.5 * this.scale;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(center.x, center.y, ladderRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Rings: reduced from 3.5 to 1.8 inches
    for (const ring of this.sim.rings) {
      const pt = this.toCanvas(ring.x, ring.y);
      const rSize = 1.8 * this.scale;
      ctx.fillStyle = ring.color === "red" ? "#e11d48" : (ring.color === "blue" ? "#2563eb" : "#facc15");
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, rSize, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#0b0f19';
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, rSize * 0.45, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Cups are neutral scoring objects: they affect placement but carry no points.
    for (const cup of (this.sim.cups || [])) {
      const pt = this.toCanvas(cup.x, cup.y);
      const rSize = 2.1 * this.scale;
      ctx.fillStyle = cup.color === "clear" ? "rgba(180,220,255,0.25)" : "#94a3b8";
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.rect(pt.x - rSize, pt.y - rSize, rSize * 2, rSize * 2);
      ctx.fill();
      ctx.stroke();
    }

    // Override goals, with alliance-colored and neutral/tall variants.
    for (const goal of this.sim.mobileGoals) {
      const pt = this.toCanvas(goal.x, goal.y);
      const goalRadius = 3.6 * this.scale;

      ctx.fillStyle = goal.color === 'neutral' ? '#eab308' : (goal.color === 'red' ? '#be123c' : '#1d4ed8');
      ctx.strokeStyle = '#f8fafc';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, goalRadius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // Center post
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 1.0 * this.scale, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  drawPlannedTrajectory() {
    const pts = this.sim.plannedSplineVisual;
    if (!pts || pts.length < 2) return;

    const ctx = this.ctx;
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();

    for (let i = 0; i < pts.length; i++) {
      const pt = this.toCanvas(pts[i].x, pts[i].y);
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Jerry.io Visual Waypoints & Path Curve
  drawJerryWaypoints() {
    const wps = this.jerryWaypoints;
    if (!wps || wps.length === 0) return;

    const ctx = this.ctx;

    // Draw connecting spline/polyline path
    if (wps.length >= 2) {
      ctx.strokeStyle = '#10b981'; // Emerald path
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      for (let i = 0; i < wps.length; i++) {
        const pt = this.toCanvas(wps[i].x, wps[i].y);
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw waypoints
    for (let i = 0; i < wps.length; i++) {
      const wp = wps[i];
      const pt = this.toCanvas(wp.x, wp.y);
      const isSelected = (i === this.selectedWpIndex);

      // Pin circle
      ctx.fillStyle = isSelected ? '#3b82f6' : '#10b981';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 11, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // Number
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px Outfit';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(i + 1, pt.x, pt.y);

      // Orientation Heading Needle
      const headRad = wp.theta * DEG_TO_RAD;
      const needleLen = 22;
      const needleX = pt.x + needleLen * Math.sin(headRad);
      const needleY = pt.y - needleLen * Math.cos(headRad);

      ctx.strokeStyle = isSelected ? '#60a5fa' : '#34d399';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(pt.x, pt.y);
      ctx.lineTo(needleX, needleY);
      ctx.stroke();

      // Needle tip handle
      ctx.fillStyle = isSelected ? '#3b82f6' : '#059669';
      ctx.beginPath();
      ctx.arc(needleX, needleY, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  drawPathTrail() {
    const trail = this.sim.pathHistory;
    if (trail.length < 2) return;

    const ctx = this.ctx;
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';

    for (let i = 1; i < trail.length; i++) {
      const p1 = this.toCanvas(trail[i - 1].x, trail[i - 1].y);
      const p2 = this.toCanvas(trail[i].x, trail[i].y);
      const speed = trail[i].v;
      const hue = clamp(180 - (speed * 110), 30, 200);
      ctx.strokeStyle = `hsla(${hue}, 85%, 55%, 0.75)`;

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }

  drawEKFEllipse() {
    const fused = this.sim.ekf.getPose();
    const pt = this.toCanvas(fused.x, fused.y);
    const ctx = this.ctx;

    const stdDevX = Math.sqrt(this.sim.ekf.P.get(0, 0)) * METER_TO_INCH * this.scale * 2.0;
    const stdDevY = Math.sqrt(this.sim.ekf.P.get(1, 1)) * METER_TO_INCH * this.scale * 2.0;

    ctx.strokeStyle = 'rgba(6, 182, 212, 0.45)';
    ctx.fillStyle = 'rgba(6, 182, 212, 0.08)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(pt.x, pt.y, Math.max(stdDevX, 4), Math.max(stdDevY, 4), 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }

  // Draw Authentic Holonomic X-Drive Robot with 45° angled omni wheels
  drawRobot() {
    const pt = this.toCanvas(this.sim.x, this.sim.y);
    const ctx = this.ctx;
    const thetaRad = this.sim.theta * DEG_TO_RAD;

    ctx.save();
    ctx.translate(pt.x, pt.y);
    ctx.rotate(thetaRad);

    const sizePx = 12.5 * this.scale;
    const half = sizePx / 2;
    const wheelW = 1.5 * this.scale;
    const wheelL = this.sim.wheelDiameterInches * this.scale * 0.9;

    // 1. Robot Base Plate
    ctx.fillStyle = '#1e293b';
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(-half, -half, sizePx, sizePx, 6);
    ctx.fill();
    ctx.stroke();

    // 2. Front Bumper Indicator (Cyan)
    ctx.fillStyle = '#06b6d4';
    ctx.fillRect(-half + 4, -half, sizePx - 8, 4);

    // 3. Intake Mechanism Animated Rollers
    ctx.fillStyle = this.sim.intakeVoltage !== 0 ? '#10b981' : '#334155';
    ctx.fillRect(-half * 0.6, -half - 3, sizePx * 0.6, 5);

    // 4. X-Drive Wheels: 4 Omni Wheels angled at 45 degrees
    const drawXWheel = (wx, wy, angleRad) => {
      ctx.save();
      ctx.translate(wx, wy);
      ctx.rotate(angleRad);

      ctx.fillStyle = '#0f172a';
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1;
      ctx.fillRect(-wheelW / 2, -wheelL / 2, wheelW, wheelL);
      ctx.strokeRect(-wheelW / 2, -wheelL / 2, wheelW, wheelL);

      ctx.strokeStyle = '#94a3b8';
      for (let s = -wheelL / 2 + 2; s < wheelL / 2; s += 4) {
        ctx.beginPath();
        ctx.moveTo(-wheelW / 2, s);
        ctx.lineTo(wheelW / 2, s);
        ctx.stroke();
      }
      ctx.restore();
    };

    // Front-Left (45 deg) & Front-Right (-45 deg)
    drawXWheel(-half + 3, -half + 3, Math.PI / 4);
    drawXWheel(half - 3, -half + 3, -Math.PI / 4);

    // Back-Left (-45 deg) & Back-Right (45 deg)
    drawXWheel(-half + 3, half - 3, -Math.PI / 4);
    drawXWheel(half - 3, half - 3, Math.PI / 4);

    // 5. Heading Arrow
    ctx.strokeStyle = '#f43f5e';
    ctx.fillStyle = '#f43f5e';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, -half * 0.9);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, -half * 1.15);
    ctx.lineTo(-4, -half * 0.85);
    ctx.lineTo(4, -half * 0.85);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }
}

// ============================================================================
// 8. Three.js 3D Viewport Engine & Onshape CAD Loader (GLTF, STL, OBJ)
// ============================================================================

// Lightweight IndexedDB storage for persistent CAD models
const CadStorage = {
  dbName: 'IRAlib_CAD_DB',
  storeName: 'cad_models',
  open() {
    return new Promise((resolve) => {
      if (!window.indexedDB) return resolve(null);
      const req = indexedDB.open(this.dbName, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(this.storeName);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
  },
  async save(key, data) {
    try {
      const db = await this.open();
      if (!db) return;
      const tx = db.transaction(this.storeName, 'readwrite');
      tx.objectStore(this.storeName).put(data, key);
    } catch (e) {
      console.warn("IndexedDB save failed:", e);
    }
  },
  async load(key) {
    try {
      const db = await this.open();
      if (!db) return null;
      return new Promise((resolve) => {
        const tx = db.transaction(this.storeName, 'readonly');
        const req = tx.objectStore(this.storeName).get(key);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
      });
    } catch (e) {
      return null;
    }
  }
};

class ThreeFieldRenderer {
  constructor(container, simulator) {
    this.container = container;
    this.sim = simulator;
    this.isActive = false;
    this.cameraMode = 'iso'; // 'iso', 'top', 'follow'
    this.cadRotationOffset = 0;
    this.cadLoaded = false;
    this.modelMode = 'cad'; // 'cad' or 'procedural'
    this.cadAutoLoadPromise = null;
    this.renderQuality = 'fast';
    this.nextRenderAt = 0;
    this.performanceWindow = {start: performance.now(), frames: 0};

    if (typeof THREE === 'undefined') {
      console.warn("Three.js not loaded.");
      return;
    }

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07090e);

    const rect = container.getBoundingClientRect();
    const aspect = (rect.width || 680) / (rect.height || 680);
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    this.camera.up.set(0,0,1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(rect.width || 680, rect.height || 680);
    this.renderer.shadowMap.enabled = false;
    this.renderer.setPixelRatio(1);
    container.appendChild(this.renderer.domElement);

    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
    }

    this.build3DField();
    this.build3DRobot();
    this.initCadLoaderUI();
    this.setCameraPreset('iso');

    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    if (!this.renderer) return;
    const rect = this.container.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    this.camera.aspect = rect.width / rect.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(rect.width, rect.height);
  }

  build3DField() {
    // Ambient & Directional Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 0.9);
    sunLight.position.set(40, -80, 100);
    sunLight.castShadow = true;
    this.scene.add(sunLight);

    const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.35);
    fillLight.position.set(-50, 60, 40);
    this.scene.add(fillLight);

    // 6x6 Field Tiles: 140.4" x 140.4" (Override)
    const fieldGeom = new THREE.PlaneGeometry(23.4, 23.4);
    const tileMaterials=[0x747a82,0x858b93].map(color=>new THREE.MeshStandardMaterial({color,roughness:.8}));
    for(let row=0;row<6;row++)for(let col=0;col<6;col++){
      const tile=new THREE.Mesh(fieldGeom,tileMaterials[(row+col)%2]);
      tile.position.set(-70.2+(col+.5)*23.4,-70.2+(row+.5)*23.4,0);
      tile.receiveShadow=true;this.scene.add(tile);
    }

    // Grid wireframe
    const grid = new THREE.GridHelper(140.4, 6, 0x384661, 0x222a3d);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.05;
    this.scene.add(grid);

    // Perimeter Wall (extruded polycarbonate look)
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x384661, transparent: true, opacity: 0.45, roughness: 0.2
    });
    const wallThick = 1.5;
    const wallHeight = 11.5;

    const makeWall = (w, h, x, y) => {
      const g = new THREE.BoxGeometry(w, h, wallHeight);
      const m = new THREE.Mesh(g, wallMat);
      m.position.set(x, y, wallHeight / 2);
      this.scene.add(m);
    };

    makeWall(140.4, wallThick, 0, 70.2);
    makeWall(140.4, wallThick, 0, -70.2);
    makeWall(wallThick, 140.4, -70.2, 0);
    makeWall(wallThick, 140.4, 70.2, 0);

    if(this.sim.override)OverrideView.build3D(this);
  }

  build3DRobot() {
    this.robot3D = new THREE.Group();

    // Container for imported Onshape CAD Model
    this.cadRobot = new THREE.Group();
    this.robot3D.add(this.cadRobot);

    // Container for Detailed Procedural VEX Override Robot Model
    this.proceduralRobot = new THREE.Group();
    this.buildDetailedProceduralRobot();
    this.robot3D.add(this.proceduralRobot);

    this.scene.add(this.robot3D);
  }

  buildDetailedProceduralRobot() {
    // Aluminum C-Channel Rails (15" x 14" chassis, brushed aluminum)
    const alumMat = new THREE.MeshStandardMaterial({ color: 0xc8d1dc, metalness: 0.85, roughness: 0.25 });
    const darkSteelMat = new THREE.MeshStandardMaterial({ color: 0x27272a, metalness: 0.7, roughness: 0.35 });
    const motorMat = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.6 });
    const greenCartridgeMat = new THREE.MeshStandardMaterial({ color: 0x22c55e, emissive: 0x15803d, emissiveIntensity: 0.3 });
    const intakeRollerMat = new THREE.MeshStandardMaterial({ color: 0x06b6d4, roughness: 0.4 });
    const pneumaticMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.9, roughness: 0.15 });

    // 1. Dual Main Longitudinal C-Channels
    const railGeom = new THREE.BoxGeometry(1.5, 14.0, 1.5);
    const leftRail = new THREE.Mesh(railGeom, alumMat);
    leftRail.position.set(-6.0, 0, 2.0);
    this.proceduralRobot.add(leftRail);

    const rightRail = new THREE.Mesh(railGeom, alumMat);
    rightRail.position.set(6.0, 0, 2.0);
    this.proceduralRobot.add(rightRail);

    // 2. Transverse Cross Rails
    const crossGeom = new THREE.BoxGeometry(10.5, 1.5, 1.5);
    const frontCross = new THREE.Mesh(crossGeom, alumMat);
    frontCross.position.set(0, 5.5, 2.0);
    this.proceduralRobot.add(frontCross);

    const backCross = new THREE.Mesh(crossGeom, alumMat);
    backCross.position.set(0, -5.5, 2.0);
    this.proceduralRobot.add(backCross);

    // 3. 4x V5 Smart Motors with Green Cartridges (45° angle mounts for X-Drive)
    const motorBoxGeom = new THREE.BoxGeometry(2.4, 1.5, 1.6);
    const cartGeom = new THREE.BoxGeometry(0.8, 1.2, 1.2);

    const addMotor = (x, y, angle) => {
      const mGroup = new THREE.Group();
      const mBody = new THREE.Mesh(motorBoxGeom, motorMat);
      const mCart = new THREE.Mesh(cartGeom, greenCartridgeMat);
      mCart.position.set(1.4, 0, 0);
      mGroup.add(mBody);
      mGroup.add(mCart);
      mGroup.position.set(x, y, 2.5);
      mGroup.rotation.z = angle;
      this.proceduralRobot.add(mGroup);
    };

    addMotor(-4.0, 4.0, Math.PI / 4);
    addMotor(4.0, 4.0, -Math.PI / 4);
    addMotor(-4.0, -4.0, -Math.PI / 4);
    addMotor(4.0, -4.0, Math.PI / 4);

    // 4. 4x Omni Wheels with Rollers at 45°
    const wheelGeom = new THREE.CylinderGeometry(2.0, 2.0, 1.4, 16);
    const wheelHubGeom = new THREE.CylinderGeometry(1.2, 1.2, 1.45, 12);
    const rollerMat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.5 });

    const add3DOmniWheel = (x, y, angle) => {
      const wGroup = new THREE.Group();
      const wRim = new THREE.Mesh(wheelGeom, rollerMat);
      wRim.rotation.y = Math.PI / 2;
      const wHub = new THREE.Mesh(wheelHubGeom, darkSteelMat);
      wHub.rotation.y = Math.PI / 2;
      wGroup.add(wRim);
      wGroup.add(wHub);
      wGroup.position.set(x, y, 2.0);
      wGroup.rotation.z = angle;
      this.proceduralRobot.add(wGroup);
    };

    add3DOmniWheel(-6.2, 6.0, Math.PI / 4);
    add3DOmniWheel(6.2, 6.0, -Math.PI / 4);
    add3DOmniWheel(-6.2, -6.0, -Math.PI / 4);
    add3DOmniWheel(6.2, -6.0, Math.PI / 4);

    // 5. Front Override Intake Mechanism (Uprights + Compliant Rollers)
    const towerGeom = new THREE.BoxGeometry(1.0, 1.0, 9.0);
    const leftTower = new THREE.Mesh(towerGeom, alumMat);
    leftTower.position.set(-3.5, 4.5, 5.5);
    this.proceduralRobot.add(leftTower);

    const rightTower = new THREE.Mesh(towerGeom, alumMat);
    rightTower.position.set(3.5, 4.5, 5.5);
    this.proceduralRobot.add(rightTower);

    // Intake Rollers (Upper & Lower)
    const rollerGeom = new THREE.CylinderGeometry(1.4, 1.4, 6.0, 16);
    const lowerRoller = new THREE.Mesh(rollerGeom, intakeRollerMat);
    lowerRoller.rotation.z = Math.PI / 2;
    lowerRoller.position.set(0, 5.0, 3.5);
    this.proceduralRobot.add(lowerRoller);

    const upperRoller = new THREE.Mesh(rollerGeom, intakeRollerMat);
    upperRoller.rotation.z = Math.PI / 2;
    upperRoller.position.set(0, 4.0, 8.5);
    this.proceduralRobot.add(upperRoller);

    // 6. Rear Override Object Mechanism (Brass cylinders + Steel hooks)
    const cylGeom = new THREE.CylinderGeometry(0.4, 0.4, 4.0, 12);
    const leftCyl = new THREE.Mesh(cylGeom, pneumaticMat);
    leftCyl.rotation.x = Math.PI / 2;
    leftCyl.position.set(-3.0, -6.5, 2.5);
    this.proceduralRobot.add(leftCyl);

    const rightCyl = new THREE.Mesh(cylGeom, pneumaticMat);
    rightCyl.rotation.x = Math.PI / 2;
    rightCyl.position.set(3.0, -6.5, 2.5);
    this.proceduralRobot.add(rightCyl);

    const clawGeom = new THREE.BoxGeometry(1.2, 3.5, 0.8);
    const leftClaw = new THREE.Mesh(clawGeom, darkSteelMat);
    leftClaw.position.set(-3.0, -8.5, 1.8);
    leftClaw.rotation.x = -0.3;
    this.proceduralRobot.add(leftClaw);

    const rightClaw = new THREE.Mesh(clawGeom, darkSteelMat);
    rightClaw.position.set(3.0, -8.5, 1.8);
    rightClaw.rotation.x = -0.3;
    this.proceduralRobot.add(rightClaw);

    // 7. V5 Robot Brain with LCD Display
    const brainBoxGeom = new THREE.BoxGeometry(4.0, 3.2, 1.2);
    const brainMat = new THREE.MeshStandardMaterial({ color: 0x09090b, roughness: 0.4 });
    const brain = new THREE.Mesh(brainBoxGeom, brainMat);
    brain.position.set(0, -1.0, 3.0);

    const screenGeom = new THREE.PlaneGeometry(3.2, 2.2);
    const screenMat = new THREE.MeshBasicMaterial({ color: 0x0284c7 });
    const screen = new THREE.Mesh(screenGeom, screenMat);
    screen.position.set(0, 0, 0.61);
    brain.add(screen);
    this.proceduralRobot.add(brain);

    // 8. V5 1100mAh Battery Pack
    const battGeom = new THREE.BoxGeometry(4.2, 1.8, 1.2);
    const battMat = new THREE.MeshStandardMaterial({ color: 0x27272a, roughness: 0.7 });
    const batt = new THREE.Mesh(battGeom, battMat);
    batt.position.set(0, -3.8, 1.5);
    this.proceduralRobot.add(batt);
  }

  initCadLoaderUI() {
    document.getElementById('renderQuality')?.addEventListener('change',event=>this.setRenderQuality(event.target.value));
    const fileInput = document.getElementById('cadFileInput');
    const btnToggleCad = document.getElementById('btnToggleCadModel');
    const btnRotateCad = document.getElementById('btnRotateCad');
    const dropOverlay = document.getElementById('cadDropOverlay');
    const wrapper = document.getElementById('canvasWrapper');

    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          this.loadCadFile(e.target.files[0]);
        }
      });
    }

    if (btnToggleCad) {
      btnToggleCad.addEventListener('click', () => {
        this.toggleModelMode();
      });
    }

    if (btnRotateCad) {
      btnRotateCad.addEventListener('click', () => {
        this.rotateCadModel();
      });
    }

    // Drag-and-Drop Handlers on Field Wrapper
    if (wrapper) {
      wrapper.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (this.isActive && dropOverlay) dropOverlay.style.display = 'flex';
      });

      wrapper.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
      });

      wrapper.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.relatedTarget && !wrapper.contains(e.relatedTarget)) {
          if (dropOverlay) dropOverlay.style.display = 'none';
        }
      });

      wrapper.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (dropOverlay) dropOverlay.style.display = 'none';
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
          this.loadCadFile(e.dataTransfer.files[0]);
        }
      });
    }
  }

  async tryAutoLoadCad() {
    for(const filename of ['robot.preview.glb','robot.glb','robot.stl']){
      try{
        const response=await fetch('models/'+filename);
        if(!response.ok)continue;
        if(Number(response.headers.get('content-length'))>RobotCAD.LIMITS.bytes){
          await response.body?.cancel();continue;
        }
        await this.loadCadFromBuffer(await response.arrayBuffer(),filename.split('.').pop(),filename);
        return;
      }catch(error){console.warn('CAD fallback:',error.message);}
    }

    try {
      const cached = await CadStorage.load('current_robot_cad');
      if (cached && cached.data) {
        await this.loadCadFromBuffer(cached.data, cached.ext, cached.name);
        return;
      }
    } catch (error) {console.warn('Cached CAD skipped:',error.message);}

    this.modelMode = 'procedural';
    this.cadRobot.visible = false;
    this.proceduralRobot.visible = true;
    const badge = document.getElementById('cadStatusBadge');
    if (badge) badge.textContent = 'CAD недоступен — процедурная модель';
  }

  ensureCadLoaded(){
    if(!this.cadAutoLoadPromise)this.cadAutoLoadPromise=this.tryAutoLoadCad();
    return this.cadAutoLoadPromise;
  }

  async loadCadFile(file) {
    try{
      RobotCAD.checkSize(file.size);
      const name=file.name,ext=name.split('.').pop().toLowerCase();
      if(!['glb','gltf','stl','obj'].includes(ext))throw Error('Поддерживаются GLB, GLTF, STL и OBJ');
      await this.ensureCadLoaded();
      const buffer=ext==='obj'?await file.text():await file.arrayBuffer();
      await this.loadCadFromBuffer(buffer,ext,name);
      await CadStorage.save('current_robot_cad',{name,data:buffer,ext});
    }catch(error){
      const badge=document.getElementById('cadStatusBadge');
      if(badge){badge.textContent=error.message;badge.title=error.message;}
    }
  }

  loadCadFromBuffer(buffer, ext, filename) {
    const badge = document.getElementById('cadStatusBadge');
    if (badge) badge.textContent = `Загрузка ${ext.toUpperCase()}...`;
    return new Promise((resolve,reject)=>{
      const failed=error=>{
        if(badge)badge.textContent='Ошибка CAD: '+error.message;
        reject(error);
      };
      const onModelReady=model=>{
        try{this.applyCadModel(model,filename);resolve();}
        catch(error){RobotCAD.disposeModel(model);failed(error);}
      };
      try{
        RobotCAD.checkSize(typeof buffer==='string'?new TextEncoder().encode(buffer).byteLength:buffer.byteLength);
        if(ext==='glb')RobotCAD.inspectGlb(buffer);
        if(ext==='glb'||ext==='gltf'){
          if(!THREE.GLTFLoader)throw Error('GLTFLoader unavailable');
          new THREE.GLTFLoader().parse(buffer,'',gltf=>onModelReady(gltf.scene||gltf.scenes[0]),failed);
        }else if(ext==='stl'){
          if(!THREE.STLLoader)throw Error('STLLoader unavailable');
          const geometry=new THREE.STLLoader().parse(buffer);
          const material=new THREE.MeshStandardMaterial({color:0x94a3b8,metalness:.75,roughness:.35});
          onModelReady(new THREE.Mesh(geometry,material));
        }else if(ext==='obj'){
          if(!THREE.OBJLoader)throw Error('OBJLoader unavailable');
          onModelReady(new THREE.OBJLoader().parse(typeof buffer==='string'?buffer:new TextDecoder().decode(buffer)));
        }else throw Error('Unsupported CAD format');
      }catch(error){failed(error);}
    });
  }

  setRenderQuality(mode){
    this.renderQuality=mode==='detail'?'detail':'fast';
    this.renderer.shadowMap.enabled=this.renderQuality==='detail';
    this.renderer.setPixelRatio(this.renderQuality==='detail'?Math.min(window.devicePixelRatio||1,2):1);
    this.nextRenderAt=0;
    this.performanceWindow={start:performance.now(),frames:0};
    this.resize();
  }

  applyCadModel(modelObject, filename) {
    const stats=RobotCAD.modelStats(modelObject);
    RobotCAD.checkCost(stats);
    const initialBox=new THREE.Box3().setFromObject(modelObject),initialSize=initialBox.getSize(new THREE.Vector3());
    const maxHorizontal=Math.max(initialSize.x,initialSize.y,initialSize.z);
    if(!Number.isFinite(maxHorizontal)||maxHorizontal<=0)throw Error('CAD has no finite geometry');
    modelObject.traverse(part=>{
      if(part.isMesh){
        part.castShadow=false;part.receiveShadow=false;
        if(!part.material)part.material=new THREE.MeshStandardMaterial({color:0x94a3b8,metalness:.6,roughness:.4});
      }
      part.updateMatrix();part.matrixAutoUpdate=false;
    });
    modelObject.scale.multiplyScalar(17.5/maxHorizontal);modelObject.updateMatrix();
    const scaledBox=new THREE.Box3().setFromObject(modelObject),scaledCenter=scaledBox.getCenter(new THREE.Vector3());
    modelObject.position.x-=scaledCenter.x;modelObject.position.y-=scaledCenter.y;modelObject.position.z-=scaledBox.min.z-.1;
    modelObject.updateMatrix();
    for(const previous of [...this.cadRobot.children]){
      this.cadRobot.remove(previous);RobotCAD.disposeModel(previous);
    }
    this.cadRobot.add(modelObject);
    this.cadLoaded=true;this.modelMode='cad';this.cadRobot.visible=true;this.proceduralRobot.visible=false;
    const button=document.getElementById('btnToggleCadModel');
    if(button)button.textContent='🤖 Режим: CAD';
    const badge=document.getElementById('cadStatusBadge');
    if(badge){
      badge.textContent=`CAD: ${filename} · ${stats.drawCalls} batches · ${Math.round(stats.triangles/1000)}k triangles`;
      badge.title='Облегчённая визуальная модель; физика и захват не берутся автоматически из CAD';
      badge.style.borderColor='#10b981';badge.style.color='#34d399';
    }
  }

  rotateCadModel() {
    this.cadRotationOffset = (this.cadRotationOffset + Math.PI / 2) % (Math.PI * 2);
    this.cadRobot.rotation.z = this.cadRotationOffset;
    this.proceduralRobot.rotation.z = this.cadRotationOffset;
  }

  toggleModelMode() {
    const btnToggleCad = document.getElementById('btnToggleCadModel');
    if (this.modelMode === 'cad') {
      this.modelMode = 'procedural';
      this.cadRobot.visible = false;
      this.proceduralRobot.visible = true;
      if (btnToggleCad) btnToggleCad.textContent = "⚙️ Режим: Процедурный";
    } else {
      this.modelMode = 'cad';
      this.cadRobot.visible = true;
      this.proceduralRobot.visible = false;
      if (btnToggleCad) btnToggleCad.textContent = "🤖 Режим: CAD";
    }
  }

  setCameraPreset(mode) {
    this.cameraMode = mode;
    const btns = ['camIso', 'camTop', 'camFollow'];
    btns.forEach(id => {
      const b = document.getElementById(id);
      if (b) b.classList.remove('active');
    });

    if (mode === 'iso') {
      document.getElementById('camIso')?.classList.add('active');
      this.camera.position.set(115, -175, 200);
      this.camera.lookAt(0, 0, 0);
      if (this.controls) this.controls.target.set(0, 0, 0);
    } else if (mode === 'top') {
      document.getElementById('camTop')?.classList.add('active');
      this.camera.position.set(0, -0.01, 205);
      this.camera.lookAt(0, 0, 0);
      if (this.controls) this.controls.target.set(0, 0, 0);
    } else if (mode === 'follow') {
      document.getElementById('camFollow')?.classList.add('active');
    }
  }

  render() {
    if (!this.isActive || !this.renderer) return;
    const now=performance.now();
    if(now<this.nextRenderAt)return;
    this.nextRenderAt=now+1000/(this.renderQuality==='fast'?30:60)-1;

    // Update 3D robot transform
    if (this.robot3D) {
      this.robot3D.position.set(this.sim.x, this.sim.y, 0);
      this.robot3D.rotation.z = -this.sim.theta * DEG_TO_RAD; // 3D counter-clockwise
    }

    if(this.sim.override)OverrideView.update3D(this);
    // Update 3D mobile goals positions dynamically
    if (this.goalMeshes && this.sim.mobileGoals) {
      for (let i = 0; i < this.goalMeshes.length && i < this.sim.mobileGoals.length; i++) {
        const goal = this.sim.mobileGoals[i];
        this.goalMeshes[i].position.set(goal.x, goal.y, 0);
      }
    }

    // Follow camera mode
    if (this.cameraMode === 'follow') {
      const offsetDist = 45;
      const angle = -this.sim.theta * DEG_TO_RAD;
      const camX = this.sim.x - offsetDist * Math.sin(angle);
      const camY = this.sim.y - offsetDist * Math.cos(angle);
      this.camera.position.lerp(new THREE.Vector3(camX, camY, 30), 0.08);
      this.camera.lookAt(this.sim.x, this.sim.y, 4);
    } else if (this.controls) {
      this.controls.update();
    }

    this.renderer.render(this.scene, this.camera);
    this.performanceWindow.frames++;
    const elapsed=now-this.performanceWindow.start;
    if(elapsed>=1000){
      const status=document.getElementById('renderPerformance'),info=this.renderer.info.render;
      if(status)status.textContent=`3D ${Math.round(this.performanceWindow.frames*1000/elapsed)} FPS · ${info.calls} draw calls · ${Math.round(info.triangles/1000)}k triangles`;
      this.performanceWindow={start:now,frames:0};
    }
  }
}

// ============================================================================
// 9. Application Bootstrap, Jerry.io Planner & UI Binding
// ============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  const canvas = document.getElementById('fieldCanvas');
  const threeContainer = document.getElementById('threeCanvasContainer');
  let core=null;
  try{core=await ProductionControl.load();}catch(error){console.error(error);}
  const playback = await window.RobotAIMotionReplay?.attach(core);
  const sim = playback?.sim ?? new VexRobotSimulator(core);
  window.nationalsSimulator=sim;
  void window.RobotAINemotron?.attach(sim);
  document.getElementById('controlStatus').textContent=core?'C++ controller ready':'C++ unavailable — use HTTP server';
  const renderer = new FieldRenderer(canvas, sim);
  let threeRenderer = null;

  try {
    threeRenderer = new ThreeFieldRenderer(threeContainer, sim);
  } catch (err) {
    console.warn("3D initialization skipped:", err);
  }

  // 2D / 3D Mode Switcher
  const tab2D = document.getElementById('tab2D');
  const tab3D = document.getElementById('tab3D');
  const camera3DControls = document.getElementById('camera3DControls');

  tab2D.addEventListener('click', () => {
    tab2D.classList.add('active');
    tab3D.classList.remove('active');
    canvas.style.display = 'block';
    threeContainer.classList.remove('active');
    camera3DControls.style.display = 'none';
    if (threeRenderer) threeRenderer.isActive = false;
  });

  tab3D.addEventListener('click', () => {
    tab3D.classList.add('active');
    tab2D.classList.remove('active');
    canvas.style.display = 'none';
    threeContainer.classList.add('active');
    camera3DControls.style.display = 'flex';
    if (threeRenderer) {
      threeRenderer.isActive = true;
      threeRenderer.performanceWindow={start:performance.now(),frames:0};
      threeRenderer.resize();
      void threeRenderer.ensureCadLoaded();
    }
  });

  // 3D Camera Controls
  document.getElementById('camIso')?.addEventListener('click', () => threeRenderer?.setCameraPreset('iso'));
  document.getElementById('camTop')?.addEventListener('click', () => threeRenderer?.setCameraPreset('top'));
  document.getElementById('camFollow')?.addEventListener('click', () => threeRenderer?.setCameraPreset('follow'));

  // Jerry.io Visual Waypoint Planner Controls
  const btnToggleJerry = document.getElementById('btnToggleJerry');
  const jerryBanner = document.getElementById('jerryBanner');
  const jerryPanel = document.getElementById('jerryPanel');
  const btnExitJerry = document.getElementById('btnExitJerry');
  const wpCountEl = document.getElementById('wpCount');
  const waypointTableBody = document.getElementById('waypointTableBody');
  const btnRunJerryPath = document.getElementById('btnRunJerryPath');
  const btnExportCpp = document.getElementById('btnExportCpp');
  const btnClearJerry = document.getElementById('btnClearJerry');

  let isJerryMode = false;

  const toggleJerryMode = (active) => {
    isJerryMode = active;
    if (isJerryMode) {
      // Force 2D view for editing
      tab2D.click();
      jerryBanner.style.display = 'flex';
      jerryPanel.classList.add('active');
      btnToggleJerry.classList.add('btn-primary');
      btnToggleJerry.classList.remove('btn-secondary');
    } else {
      jerryBanner.style.display = 'none';
      jerryPanel.classList.remove('active');
      btnToggleJerry.classList.remove('btn-primary');
      btnToggleJerry.classList.add('btn-secondary');
    }
  };

  btnToggleJerry.addEventListener('click', () => toggleJerryMode(!isJerryMode));
  btnExitJerry.addEventListener('click', () => toggleJerryMode(false));

  const syncWaypointTable = () => {
    const wps = renderer.jerryWaypoints;
    wpCountEl.textContent = wps.length;

    if (wps.length === 0) {
      waypointTableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; color: var(--text-dim); padding: 1rem;">
            Кликните по полю, чтобы поставить первую точку!
          </td>
        </tr>`;
      return;
    }

    waypointTableBody.innerHTML = '';
    wps.forEach((wp, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="wp-num-badge">${idx + 1}</span></td>
        <td>${wp.x.toFixed(1)}"</td>
        <td>${wp.y.toFixed(1)}"</td>
        <td>${wp.theta.toFixed(0)}°</td>
        <td>
          <select data-idx="${idx}" class="wp-type-select" style="background: rgba(255,255,255,0.08); color: white; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; padding: 2px 4px; font-size: 0.75rem;">
            <option value="spline" ${wp.type === 'spline' ? 'selected' : ''}>Spline (Кривая)</option>
            <option value="strafe" ${wp.type === 'strafe' ? 'selected' : ''}>Strafe (Боком)</option>
            <option value="drive" ${wp.type === 'drive' ? 'selected' : ''}>Drive (Прямо)</option>
            <option value="turn" ${wp.type === 'turn' ? 'selected' : ''}>Turn (Разворот)</option>
          </select>
        </td>
        <td>${(wp.speed || 1.0).toFixed(1)}x</td>
        <td style="text-align: right;">
          <button data-idx="${idx}" class="wp-action-btn btn-del-wp" title="Удалить">✕</button>
        </td>
      `;
      waypointTableBody.appendChild(tr);
    });

    // Bind type dropdowns & delete buttons
    document.querySelectorAll('.wp-type-select').forEach(sel => {
      sel.addEventListener('change', (e) => {
        const i = parseInt(e.target.dataset.idx);
        wps[i].type = e.target.value;
      });
    });

    document.querySelectorAll('.btn-del-wp').forEach(b => {
      b.addEventListener('click', (e) => {
        const i = parseInt(e.target.dataset.idx);
        wps.splice(i, 1);
        syncWaypointTable();
      });
    });
  };

  // Canvas Mouse Interactions for Jerry.io Planner
  canvas.addEventListener('mousedown', (e) => {
    if (!isJerryMode) return;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    const fieldPt = renderer.toField(px, py);

    // Check click on existing waypoint
    let clickedWp = -1;
    let clickedHeading = false;

    for (let i = 0; i < renderer.jerryWaypoints.length; i++) {
      const wp = renderer.jerryWaypoints[i];
      const pt = renderer.toCanvas(wp.x, wp.y);
      const distPin = Math.hypot(px - pt.x, py - pt.y);

      // Check heading needle tip
      const headRad = wp.theta * DEG_TO_RAD;
      const needleX = pt.x + 22 * Math.sin(headRad);
      const needleY = pt.y - 22 * Math.cos(headRad);
      const distNeedle = Math.hypot(px - needleX, py - needleY);

      if (distNeedle < 10) {
        clickedWp = i;
        clickedHeading = true;
        break;
      } else if (distPin < 14) {
        clickedWp = i;
        break;
      }
    }

    if (clickedWp >= 0) {
      renderer.selectedWpIndex = clickedWp;
      renderer.draggedWpIndex = clickedWp;
      renderer.dragMode = clickedHeading ? 'heading' : 'pos';
    } else {
      // Add new waypoint
      const newWp = {
        id: renderer.jerryWaypoints.length + 1,
        x: clamp(fieldPt.x, -70, 70),
        y: clamp(fieldPt.y, -70, 70),
        theta: 0,
        type: renderer.jerryWaypoints.length === 0 ? 'drive' : 'spline',
        speed: 1.0
      };
      renderer.jerryWaypoints.push(newWp);
      renderer.selectedWpIndex = renderer.jerryWaypoints.length - 1;
      syncWaypointTable();
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!isJerryMode || renderer.draggedWpIndex < 0) return;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    const wp = renderer.jerryWaypoints[renderer.draggedWpIndex];

    if (renderer.dragMode === 'pos') {
      const fieldPt = renderer.toField(px, py);
      wp.x = clamp(fieldPt.x, -70, 70);
      wp.y = clamp(fieldPt.y, -70, 70);
      syncWaypointTable();
    } else if (renderer.dragMode === 'heading') {
      const pt = renderer.toCanvas(wp.x, wp.y);
      const dx = px - pt.x;
      const dy = -(py - pt.y);
      wp.theta = Math.atan2(dx, dy) * RAD_TO_DEG;
      syncWaypointTable();
    }
  });

  window.addEventListener('mouseup', () => {
    renderer.draggedWpIndex = -1;
    renderer.dragMode = 'none';
  });

  btnClearJerry.addEventListener('click', () => {
    renderer.jerryWaypoints = [];
    renderer.selectedWpIndex = -1;
    syncWaypointTable();
  });

  // Run Custom Jerry.io Waypoint Route
  btnRunJerryPath.addEventListener('click', () => {
    const wps = renderer.jerryWaypoints;
    if (wps.length === 0) return;

    sim.resetSimulation();
    sim.isRunning = true;
    sim.setPose(wps[0].x, wps[0].y, wps[0].theta);

    for (let i = 1; i < wps.length; i++) {
      const prev = wps[i - 1];
      const cur = wps[i];

      if (cur.type === 'spline') {
        sim.queueAction({
          type: 'spline',
          start: { x: prev.x, y: prev.y, theta: prev.theta },
          end: { x: cur.x, y: cur.y, theta: cur.theta },
          maxVel: 1.0 * cur.speed, maxAccel: 1.8,
          desc: `WP ${i + 1}: Spline to (${cur.x.toFixed(0)}, ${cur.y.toFixed(0)})`
        });
      } else if (cur.type === 'strafe') {
        sim.queueAction({type:'pose',targetX:cur.x,targetY:cur.y,targetTheta:cur.theta,desc:`WP ${i+1}: strafe`});
      } else if (cur.type === 'turn') {
        sim.queueAction({
          type: 'turn',
          targetHeading: cur.theta,
          desc: `WP ${i + 1}: Snap Turn ${cur.theta.toFixed(0)}deg`
        });
      } else {
        sim.queueAction({
          type: 'drivePoint',
          targetX: cur.x, targetY: cur.y,
          desc: `WP ${i + 1}: Drive to (${cur.x.toFixed(0)}, ${cur.y.toFixed(0)})`
        });
      }
    }
  });

  // Export C++ Code Modal
  const exportModal = document.getElementById('exportModal');
  const cppCodePreview = document.getElementById('cppCodePreview');
  const btnModalClose = document.getElementById('btnModalClose');
  const btnModalDismiss = document.getElementById('btnModalDismiss');
  const btnCopyCode = document.getElementById('btnCopyCode');

  btnExportCpp.addEventListener('click', () => {
    const wps = renderer.jerryWaypoints;
    let code = `/**\n * @brief Autonomous Routine Generated by IRAlib Jerry.io Planner\n */\nvoid autoJerryCustomRoutine() {\n`;

    if (wps.length === 0) {
      code += `    // Нет путевых точек. Добавьте точки на поле!\n`;
    } else {
      code += `    chassis.setPose(${wps[0].x.toFixed(1)}, ${wps[0].y.toFixed(1)}, ${wps[0].theta.toFixed(1)});\n\n`;

      for (let i = 1; i < wps.length; i++) {
        const prev = wps[i - 1];
        const cur = wps[i];

        const x=cur.type==='turn'?prev.x:cur.x,y=cur.type==='turn'?prev.y:cur.y;
        const call=cur.type==='spline'?'autoSpline':'autoPose';
        code += `    if (${call}(${x.toFixed(1)}, ${y.toFixed(1)}, ${cur.theta.toFixed(1)}) != lemlib::MotionResult::Settled) return;\n`;

      }
    }
    code += `    controller.print(0, 0, "Custom Path Done!");\n}`;

    cppCodePreview.textContent = code;
    exportModal.classList.add('active');
  });

  const closeModal = () => exportModal.classList.remove('active');
  btnModalClose.addEventListener('click', closeModal);
  btnModalDismiss.addEventListener('click', closeModal);

  btnCopyCode.addEventListener('click', () => {
    navigator.clipboard.writeText(cppCodePreview.textContent);
    btnCopyCode.textContent = "Скопировано!";
    setTimeout(() => { btnCopyCode.textContent = "Скопировать в буфер"; }, 1500);
  });

  // UI Element References
  const routineSelect = document.getElementById('routineSelect');
  const btnPlay = document.getElementById('btnPlay');
  const btnPause = document.getElementById('btnPause');
  const btnReset = document.getElementById('btnReset');
  const btnOverrideStart = document.getElementById('btnOverrideStart');
  const btnOverrideReset = document.getElementById('btnOverrideReset');
  const btnStep = document.getElementById('btnStep');
  const speedButtons = document.querySelectorAll('.speed-opt');

  const brainLcdLines = [
    document.getElementById('brainLcd0'),
    document.getElementById('brainLcd1'),
    document.getElementById('brainLcd2'),
    document.getElementById('brainLcd3')
  ];

  const ctrlLcdLines = [
    document.getElementById('ctrlLcd0'),
    document.getElementById('ctrlLcd1'),
    document.getElementById('ctrlLcd2')
  ];

  const controllerShell = document.getElementById('controllerShell');

  const statX = document.getElementById('statX');
  const statY = document.getElementById('statY');
  const statTheta = document.getElementById('statTheta');
  const statVel = document.getElementById('statVel');
  const statLeftVolt = document.getElementById('statLeftVolt');
  const statRightVolt = document.getElementById('statRightVolt');

  const chartVel = document.getElementById('chartVel');
  const ctxVel = chartVel ? chartVel.getContext('2d') : null;

  btnPlay.addEventListener('click', () => {
    if (playback?.finished) playback.restart();
    if (!playback && !sim.matchMode && !sim.isRunning && sim.routineQueue.length === 0) {
      sim.startRoutine(routineSelect.value);
    }
    sim.isPaused = false;
  });

  btnPause.addEventListener('click', () => {
    sim.isPaused = !sim.isPaused;
  });

  btnReset.addEventListener('click', () => {
    if (playback) playback.restart();
    else sim.resetSimulation();
  });

  btnOverrideStart?.addEventListener('click', () => {
    const choices=Object.fromEntries([...document.querySelectorAll('[data-auto-robot]')].map(el=>[el.dataset.autoRobot,el.value]));
    gameUI.stopLift();
    const mode=document.getElementById('gameMode').value;
    const result=OverrideUI.startGame(sim,mode,choices,{preload:document.getElementById('practicePreload').checked});
    document.getElementById('gameFeedback').textContent=result.ok?(mode==='match'?'Матч начался: автономки работают первые 15 секунд.':'Игра началась: WASD — движение, E/Q — подъёмник, F — взять / Toggle.'):result.error;
  });

  btnOverrideReset?.addEventListener('click', () => {
    if (sim.override) {
      sim.resetSimulation();
      sim.triggerRumble(".");
    }
  });

  btnStep.addEventListener('click', () => {
    sim.isPaused = false;
    if (playback) playback.step(0.01);
    else sim.update(0.01);
    sim.isPaused = true;
  });

  speedButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      speedButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      sim.timeScale = parseFloat(btn.dataset.speed || '0.5');
    });
  });

  document.getElementById('toggleTarget')?.addEventListener('change', (e) => { renderer.showTargetPath = e.target.checked; });
  document.getElementById('toggleTrail')?.addEventListener('change', (e) => { renderer.showTrail = e.target.checked; });
  document.getElementById('toggleEKF')?.addEventListener('change', (e) => { renderer.showEKF = e.target.checked; });
  document.getElementById('toggleCoord')?.addEventListener('change', (e) => { renderer.showCoordinates = e.target.checked; });

  const bindCtrlBtn = (id, callback) => {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.addEventListener('click', () => {
      if (playback) return;
      btn.classList.add('pressed');
      setTimeout(() => btn.classList.remove('pressed'), 120);
      callback();
    });
  };

  bindCtrlBtn('btnCtrlA', () => {if(sim.fleet)for(const e of sim.fleet.entries.values())sim.fleet.stop(e,'Cancelled');sim.isRunning=false;sim.routineQueue=[];sim.currentAction=null;sim.commandedWheelVoltages=null;sim.motorVolts=[0,0,0,0];sim.holonomicArcade(0,0,0);sim.lastMotionResult='Cancelled';});
  bindCtrlBtn('btnCtrlB', () => sim.startRoutine('testLinearDrive'));
  bindCtrlBtn('btnCtrlY', () => sim.startRoutine('testAngularTurn'));
  bindCtrlBtn('btnCtrlX', () => sim.startRoutine('testStrafe'));
  bindCtrlBtn('btnCtrlUP', () => sim.startRoutine('testQuinticSpline'));
  bindCtrlBtn('btnCtrlRIGHT', () => sim.startRoutine('testBezier'));
  bindCtrlBtn('btnCtrlDOWN', () => sim.startRoutine('testDiagonal'));
  bindCtrlBtn('btnCtrlL1', () => sim.startRoutine('autoHolonomicSkills'));

  document.getElementById('gameRobot')?.addEventListener('change',e=>{
    if(!sim.matchMode)return;
    sim.activeRobotId=e.target.value;sim.holonomicArcade(0,0,0);sim.fleet?.syncView();
  });
  const gameUI=OverrideUI.attach(sim,{replay:!!playback});
  NationalsTacticsUI.attach(sim,{replay:!!playback});
  document.getElementById('addRuling')?.addEventListener('click',()=>{
    const result=sim.override.adjudicate(sim.activeRobotId,document.getElementById('refRule').value,document.getElementById('refSeverity').value,document.getElementById('refReason').value,{autonomous:document.getElementById('refAuto').checked,awardAWP:document.getElementById('refAWP').checked});
    document.getElementById('gameFeedback').textContent=result.ok?'Решение записано':result.error;
  });
  document.getElementById('exportMatch')?.addEventListener('click',()=>{
    if(!sim.fleet)return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(sim.fleet.report(),null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='nationals-match.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  // Keyboard Controls (Holonomic 3-DOF: W/S forward/backward, Q/E or A/D strafe, ArrowLeft/ArrowRight turn)
  const keysDown = {};
  const keyboardKey=event=>(event.code?.startsWith('Key')?event.code.slice(3):event.key||'').toLowerCase();
  window.addEventListener('keydown', (event) => {
    if(event.ctrlKey||event.altKey||event.metaKey||event.target?.matches?.('input,select,textarea,[contenteditable]'))return;
    keysDown[keyboardKey(event)] = true;
    if (event.key === ' '&&!event.repeat) {
      sim.isPaused = !sim.isPaused;
      gameUI.stopLift();event.preventDefault();
    }
  });
  window.addEventListener('keyup', (event) => {
    delete keysDown[keyboardKey(event)];
  });

  window.addEventListener('blur',()=>{for(const key of Object.keys(keysDown))delete keysDown[key];sim.holonomicArcade(0,0,0);});

  // Animation Loop with delta accumulator for exact real-time speed & smooth playback
  let lastFrameTime = performance.now();
  let physicsAccumulator = 0;
  const PHYSICS_DT = 0.01; // 100 Hz fixed physics step

  function loop(currentTime) {
    if (!currentTime) currentTime = performance.now();
    let deltaSeconds = (currentTime - lastFrameTime) / 1000.0;
    lastFrameTime = currentTime;

    // Clamp to prevent lag jumps
    if (deltaSeconds > 0.1) deltaSeconds = 0.1;

    // Manual Holonomic 3-DOF Drive from Keyboard
    if (!playback && !sim.isRunning && !isJerryMode) {
      let forward = 0;
      let strafe = 0;
      let turn = 0;

      if (keysDown['w'] || keysDown['arrowup']) forward += 1.0;
      if (keysDown['s'] || keysDown['arrowdown']) forward -= 1.0;
      if (keysDown['d'] || (!sim.matchMode&&keysDown['e'])) strafe += 1.0;
      if (keysDown['a'] || (!sim.matchMode&&keysDown['q'])) strafe -= 1.0;
      if (keysDown['arrowright']) turn += 0.8;
      if (keysDown['arrowleft']) turn -= 0.8;

      const drivePower=sim.matchMode?Number(document.getElementById('gameDrivePower').value):1;
      sim.holonomicArcade(forward*drivePower, strafe*drivePower, turn*drivePower);
    } else {
      sim.holonomicArcade(0, 0, 0);
    }

    // Accumulate time scaled by selected speed
    physicsAccumulator += deltaSeconds * sim.timeScale;

    // Step physics at fixed rate
    let substeps = 0;
    const MAX_SUBSTEPS = 15;
    while (physicsAccumulator >= PHYSICS_DT && substeps < MAX_SUBSTEPS) {
      if (playback) playback.step(PHYSICS_DT);
      else {gameUI.stepControls(PHYSICS_DT);sim.update(PHYSICS_DT);}
      physicsAccumulator -= PHYSICS_DT;
      substeps++;
    }
    if (substeps >= MAX_SUBSTEPS) {
      physicsAccumulator = 0;
    }

    // Render active view (2D or 3D)
    if (!threeRenderer || !threeRenderer.isActive) {
      renderer.render();
    } else {
      threeRenderer.render();
    }

    gameUI.update();
    // Update Override scoreboard and telemetry UI.
    if (sim.override) {
      const overrideState = sim.override.getState();
      const phase = document.getElementById("overridePhase");
      const clock = document.getElementById("overrideClock");
      const red = document.getElementById("overrideRed");
      const blue = document.getElementById("overrideBlue");
      if (phase) phase.textContent = overrideState.mode==='practice'&&!overrideState.matchEnded?'PRACTICE':overrideState.phase.replace("_", " ").toUpperCase();
      if (clock) {
        const remaining = Math.max(0, overrideState.rules.matchSeconds - overrideState.clock);
        clock.textContent = overrideState.mode==='practice'?'∞':Math.floor(remaining / 60) + ":" + String(Math.floor(remaining % 60)).padStart(2, "0");
      }
      document.getElementById('ruleLog').textContent=overrideState.ruleEvents.slice(-8).map(e=>`${e.clock.toFixed(1)} с · ${e.robotId} · ${e.rule} · ${e.severity}: ${e.detail}`).join(' | ');
      const activeRobot=sim.override.robots.find(r=>r.id===sim.activeRobotId);
      document.getElementById('liftStatus').textContent=`Подъёмник ${activeRobot.manipulator.height.toFixed(1)}″ → ${activeRobot.manipulator.target.toFixed(1)}″`;
      document.getElementById('matchStatus').textContent=overrideState.matchEnded?(overrideState.scoreFinal?'Счёт зафиксирован':`Ожидание остановки: ${overrideState.settlingSeconds.toFixed(1)} / 5 с`):sim.fleet?[...sim.fleet.entries].map(([id,e])=>`${id}: ${e.status}${e.error?' ('+e.error+')':''}`).join(' · '):'';
      if (red) red.textContent = String(overrideState.score.red);
      if (blue) blue.textContent = String(overrideState.score.blue);
    }    // Update Telemetry UI
    statX.textContent = `${sim.x.toFixed(1)}"`;
    statY.textContent = `${sim.y.toFixed(1)}"`;
    statTheta.textContent = `${sim.theta.toFixed(1)}°`;
    statVel.textContent = `${sim.v.toFixed(2)} m/s`;
    statLeftVolt.textContent = `${((sim.actionThrottle || 0) * 12).toFixed(1)} V`;
    statRightVolt.textContent = `${((sim.actionStrafe || 0) * 12).toFixed(1)} V`;

    for (let i = 0; i < 4; i++) {
      if (brainLcdLines[i]) brainLcdLines[i].textContent = sim.brainLcdLines[i];
    }

    for (let i = 0; i < 3; i++) {
      if (ctrlLcdLines[i]) ctrlLcdLines[i].textContent = sim.controllerLcdLines[i] || "";
    }

    if (sim.rumbleActive) {
      controllerShell?.classList.add('rumble-active');
    } else {
      controllerShell?.classList.remove('rumble-active');
    }

    if (ctxVel && sim.telemetry.time.length > 2) {
      const w = chartVel.width;
      const h = chartVel.height;
      ctxVel.clearRect(0, 0, w, h);

      ctxVel.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctxVel.beginPath();
      ctxVel.moveTo(0, h / 2);
      ctxVel.lineTo(w, h / 2);
      ctxVel.stroke();

      ctxVel.strokeStyle = '#06b6d4';
      ctxVel.lineWidth = 1.5;
      ctxVel.beginPath();
      for (let i = 0; i < sim.telemetry.vActual.length; i++) {
        const x = (i / 150) * w;
        const y = (h / 2) - (sim.telemetry.vActual[i] / 1.5) * (h / 2);
        if (i === 0) ctxVel.moveTo(x, y);
        else ctxVel.lineTo(x, y);
      }
      ctxVel.stroke();
    }

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
});
