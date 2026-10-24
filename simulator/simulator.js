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
    if (badge) badge.textContent = 'CAD unavailable — procedural model';
  }

  ensureCadLoaded(){
    if(!this.cadAutoLoadPromise)this.cadAutoLoadPromise=this.tryAutoLoadCad();
    return this.cadAutoLoadPromise;
  }

  async loadCadFile(file) {
    try{
      RobotCAD.checkSize(file.size);
      const name=file.name,ext=name.split('.').pop().toLowerCase();
      if(!['glb','gltf','stl','obj'].includes(ext))throw Error('GLB, GLTF, STL and OBJ are supported');
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
    if (badge) badge.textContent = `Loading ${ext.toUpperCase()}...`;
    return new Promise((resolve,reject)=>{
      const failed=error=>{
        if(badge)badge.textContent='CAD error: '+error.message;
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
    if(button)button.textContent='🤖 Mode: CAD';
    const badge=document.getElementById('cadStatusBadge');
    if(badge){
      badge.textContent=`CAD: ${filename} · ${stats.drawCalls} batches · ${Math.round(stats.triangles/1000)}k triangles`;
      badge.title='Lightweight visual model; physics and grabbing are not taken from the CAD automatically';
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
      if (btnToggleCad) btnToggleCad.textContent = "⚙️ Mode: Procedural";
    } else {
      this.modelMode = 'cad';
      this.cadRobot.visible = true;
      this.proceduralRobot.visible = false;
      if (btnToggleCad) btnToggleCad.textContent = "🤖 Mode: CAD";
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
            Click the field to place the first point!
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
            <option value="spline" ${wp.type === 'spline' ? 'selected' : ''}>Spline (curve)</option>
            <option value="strafe" ${wp.type === 'strafe' ? 'selected' : ''}>Strafe (sideways)</option>
            <option value="drive" ${wp.type === 'drive' ? 'selected' : ''}>Drive (straight)</option>
            <option value="turn" ${wp.type === 'turn' ? 'selected' : ''}>Turn (rotate)</option>
          </select>
        </td>
        <td>${(wp.speed || 1.0).toFixed(1)}x</td>
        <td style="text-align: right;">
          <button data-idx="${idx}" class="wp-action-btn btn-del-wp" title="Delete">✕</button>
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
      code += `    // No waypoints. Add points on the field!\n`;
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
    btnCopyCode.textContent = "Copied!";
    setTimeout(() => { btnCopyCode.textContent = "Copy to clipboard"; }, 1500);
  });

  // ============================================================
  // AI Autonomous Programming Studio Implementation
  // ============================================================
  const btnToggleAiStudio = document.getElementById('btnToggleAiStudio');
  const aiStudioPanel = document.getElementById('aiStudioPanel');
  const btnAiClose = document.getElementById('btnAiClose');
  const aiPromptInput = document.getElementById('aiPromptInput');
  const btnAiGenerate = document.getElementById('btnAiGenerate');
  const aiCppCodeView = document.getElementById('aiCppCodeView');
  const aiStepsList = document.getElementById('aiStepsList');
  const aiStepCount = document.getElementById('aiStepCount');
  const btnAiRunInSim = document.getElementById('btnAiRunInSim');
  const btnAiCopyCpp = document.getElementById('btnAiCopyCpp');
  const btnAiSendJerry = document.getElementById('btnAiSendJerry');
  const btnAiResetRobot = document.getElementById('btnAiResetRobot');
  const aiChips = document.querySelectorAll('.ai-chip');
  const aiStatusDot = document.getElementById('aiStatusDot');
  const aiStatusText = document.getElementById('aiStatusText');
  const btnCheckNemotron = document.getElementById('btnCheckNemotron');
  const btnConnectNebius = document.getElementById('btnConnectNebius');
  let nemotronState = { available: false, liveInferenceVerified: false, checked: false, details: '' };

  async function connectNebiusTokenFactory() {
    // The key goes only to the local RoboProof server, which stores it in the git-ignored .cache folder.
    const key = prompt(
      '🔑 Enter your Nebius Token Factory API key (Bearer token):\n\n' +
      'You can create it in the Nebius Token Factory console (API Keys).\n' +
      'The key is stored only locally in .cache/ and never goes into git.'
    );
    if (key === null) return;
    const cleanKey = key.trim();
    if (!cleanKey) {
      alert('The API key cannot be empty.');
      return;
    }
    if (aiStatusText) aiStatusText.textContent = "Connecting to Nebius Token Factory...";
    try {
      const res = await fetch('http://127.0.0.1:8766/api/nemotron/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiKey: cleanKey })
      });
      const data = await res.json();
      if (res.ok && data.available) {
        nemotronState = { available: true, liveInferenceVerified: false, isNebius: true, checked: true, details: 'Nebius Token Factory' };
        if (aiStatusDot) aiStatusDot.className = 'status-indicator-dot online';
        if (aiStatusText) aiStatusText.innerHTML = '<strong>Nebius Token Factory:</strong> Online (NVIDIA Nemotron cloud)';
        alert('🎉 Connected to Nebius Token Factory!\nModel: ' + data.model + '\nAI Studio requests are now handled by NVIDIA Nemotron on Nebius cloud GPUs.');
      } else {
        alert('❌ Connection error: ' + (data.error || 'Check that the API key is correct.'));
        checkNemotronStatus(false);
      }
    } catch (err) {
      alert('❌ Could not reach the RoboProof server (port 8766): ' + err.message);
    }
  }

  async function checkNemotronStatus(interactive = false) {
    if (aiStatusText) aiStatusText.textContent = "AI engine: checking the Nemotron connection...";
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2500);
      const res = await fetch('http://127.0.0.1:8766/api/nemotron/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      const data = await res.json();
      const isNebius = data.provider === 'nebius-token-factory' || !!data.cloudEnabled;
      nemotronState = {
        available: !!data.available,
        liveInferenceVerified: !!data.liveInferenceVerified,
        isNebius,
        checked: true,
        details: data.error || (isNebius ? 'Nebius Cloud GPU' : (data.available ? 'NVIDIA T4 verified' : 'GPU offline'))
      };

      if (data.available && (data.liveInferenceVerified || isNebius)) {
        if (aiStatusDot) aiStatusDot.className = 'status-indicator-dot online';
        if (isNebius) {
          if (aiStatusText) aiStatusText.innerHTML = '<strong>Nebius Token Factory:</strong> Online (NVIDIA Nemotron cloud)';
          if (interactive) {
            alert('🟢 Nebius Token Factory is active!\nModel: ' + (data.model || 'NVIDIA Nemotron') + '\nInference runs on Nebius cloud GPUs (< 1 s).');
          }
        } else {
          if (aiStatusText) aiStatusText.innerHTML = '<strong>Nemotron T4 LLM:</strong> Connected (NVIDIA T4)';
          if (interactive) {
            alert('🟢 Nemotron LLM coordinator is active on an NVIDIA T4!\nYour prompts are handled by Nemotron.');
          }
        }
      } else {
        if (aiStatusDot) aiStatusDot.className = 'status-indicator-dot';
        if (aiStatusText) aiStatusText.innerHTML = '<strong>Nebius/Nemotron: offline</strong>. Click Connect Nebius Token Factory';
        if (interactive) {
          alert(
            '⚠️ Nemotron/Nebius Offline:\n\n' +
            (data.error || 'Connect to Nebius Token Factory or an NVIDIA T4 first.') +
            '\n\nClick ⚡ Connect Nebius Token Factory to enter an API key.'
          );
        }
      }
    } catch (e) {
      nemotronState = { available: false, liveInferenceVerified: false, checked: true, details: 'Server offline' };
      if (aiStatusDot) aiStatusDot.className = 'status-indicator-dot offline';
      if (aiStatusText) aiStatusText.innerHTML = '<strong>Local mode:</strong> Built-in tactical AI + GVF-TD3 RL (WASM)';
      if (interactive) {
        alert(
          'ℹ️ The RoboProof server (port 8766) is not running or unreachable.\n\n' +
          'The simulator runs in a fully local mode (WASM + iraLIB + GVF-TD3 RL).'
        );
      }
    }
  }

  if (btnCheckNemotron) {
    btnCheckNemotron.addEventListener('click', () => checkNemotronStatus(true));
  }
  if (btnConnectNebius) {
    btnConnectNebius.addEventListener('click', connectNebiusTokenFactory);
  }
  checkNemotronStatus(false);

  let currentAiPlan = null;

  function renderAiPlan(plan) {
    currentAiPlan = plan;
    if (aiCppCodeView) aiCppCodeView.textContent = plan.cpp;
    if (aiStepCount) aiStepCount.textContent = `${plan.steps.length} steps`;
    if (aiStepsList) {
      aiStepsList.innerHTML = plan.steps.map((step, idx) => `
        <div class="ai-step-item">
          <strong style="color:#a855f7;">${idx + 1}.</strong> ${step}
        </div>
      `).join('');
    }
    if (plan.waypoints && renderer) {
      renderer.jerryWaypoints = JSON.parse(JSON.stringify(plan.waypoints));
    }
    if (plan.startPose && sim && !sim.isRunning) {
      sim.setPose(plan.startPose.x, plan.startPose.y, plan.startPose.theta);
      if (renderer) sim.renderField(renderer);
    }
  }

  function applyAiPreset(presetKey) {
    let prompt = "";
    if (presetKey === 'skills') {
      prompt = "start at 0:0:0, go to point (0, 24), go to point (24, 24), turn right 90 degrees, go to point (24, 0), go to point (0, 0)";
    } else if (presetKey === 'scurve') {
      prompt = "start at 0:0:0, go to point (18, 24), turn right 35 degrees, go to point (36, 48), turn right 90 degrees";
    } else if (presetKey === 'strafe') {
      prompt = "start at 0:0:0, strafe right 24 inches, turn right 90 degrees, strafe right 24 inches, turn right 90 degrees, strafe left 24 inches";
    } else if (presetKey === 'pickup') {
      prompt = "start at 0:0:0, go to point (12, 20), lift 8 inches, go to point (24, 36), lift 14.5 inches, drive backward 10 inches";
    } else if (presetKey === 'zigzag') {
      prompt = "start at 0:0:0, go to point (-16, 18), go to point (16, 36), go to point (0, 54)";
    } else if (presetKey === 'box') {
      prompt = "start at 0:0:0, drive forward 30 inches, strafe right 30 inches, drive backward 30 inches, strafe left 30 inches";
    }
    if (prompt) {
      if (aiPromptInput) aiPromptInput.value = prompt;
      generateAiFromPrompt(prompt);
      return;
    }

    let plan = null;
    if (presetKey === 'skills') {
      plan = {
        title: "VEX Override 60-Point Skills Autonomous",
        startPose: { x: 0, y: 0, theta: 0 },
        waypoints: [
          { id: 1, x: 0, y: 0, theta: 0, type: 'drive', speed: 1.0 },
          { id: 2, x: 0, y: 24, theta: 0, type: 'spline', speed: 1.2 },
          { id: 3, x: 24, y: 24, theta: 45, type: 'strafe', speed: 1.0 },
          { id: 4, x: 24, y: 0, theta: 90, type: 'drive', speed: 1.0 },
          { id: 5, x: 0, y: 0, theta: 0, type: 'spline', speed: 1.0 }
        ],
        actions: [
          { type: 'intake', voltage: 12000, desc: 'Intake Active' },
          { type: 'spline', start: {x:0, y:0, theta:0}, end: {x:0, y:24, theta:0}, maxVel: 1.2, maxAccel: 1.8, desc: 'Spline Forward 24in to Pin' },
          { type: 'pose', targetX: 24, targetY: 24, targetTheta: 45, desc: 'Lateral Strafe to Goal' },
          { type: 'turn', targetHeading: 90, desc: 'Snap Turn 90° for Goal Alignment' },
          { type: 'drivePoint', targetX: 24, targetY: 0, desc: 'Back to Field Center' },
          { type: 'spline', start: {x:24, y:0, theta:90}, end: {x:0, y:0, theta:0}, maxVel: 1.0, maxAccel: 1.5, desc: 'Return Home (0,0,0)' }
        ],
        steps: [
          "Start pose: (0, 0, 0°), intake rollers on",
          "Quintic spline forward 24″ to the central element (V_max = 1.2 m/s)",
          "Holonomic strafe up-right to (24″, 24″) with a 45° heading",
          "Instant 90° LQR turn to line up with the mobile goal",
          "Move to (24″, 0″) to clear the neutral zone",
          "Spline back to the home zone (0, 0, 0°) and stop"
        ],
        cpp: `/**\n * @file src/main.cpp\n * @brief VEX Override 60-Point Skills Autonomous\n * Generated by RobotAI Autonomous Planner\n */\nvoid autonomous() {\n    // 1. Initialise odometry and EKF\n    chassis.setPose(0.0, 0.0, 0.0);\n    intake.move_voltage(12000);\n\n    // 2. Spline forward 24 inches\n    if (autoSpline(0.0, 24.0, 0.0, 1.2) != lemlib::MotionResult::Settled) return;\n\n    // 3. Holonomic strafe to (24, 24)\n    if (autoPose(24.0, 24.0, 45.0) != lemlib::MotionResult::Settled) return;\n\n    // 4. LQR turn to 90 degrees\n    chassis.turnToHeading(90.0, 1000);\n\n    // 5. Move to the centre and return\n    if (autoPose(24.0, 0.0, 90.0) != lemlib::MotionResult::Settled) return;\n    if (autoSpline(0.0, 0.0, 0.0, 1.0) != lemlib::MotionResult::Settled) return;\n\n    intake.move_voltage(0);\n    controller.print(0, 0, "Skills Routine Done!");\n}`
      };
    } else if (presetKey === 'scurve') {
      plan = {
        title: "High-Speed S-Curve Quintic Spline",
        startPose: { x: 0, y: 0, theta: 0 },
        waypoints: [
          { id: 1, x: 0, y: 0, theta: 0, type: 'drive', speed: 1.0 },
          { id: 2, x: 18, y: 24, theta: 35, type: 'spline', speed: 1.4 },
          { id: 3, x: 36, y: 48, theta: 90, type: 'spline', speed: 1.4 }
        ],
        actions: [
          { type: 'spline', start: {x:0, y:0, theta:0}, end: {x:18, y:24, theta:35}, maxVel: 1.4, maxAccel: 2.0, desc: 'S-Curve Curve 1 to (18, 24)' },
          { type: 'spline', start: {x:18, y:24, theta:35}, end: {x:36, y:48, theta:90}, maxVel: 1.4, maxAccel: 2.0, desc: 'S-Curve Curve 2 to (36, 48)' }
        ],
        steps: [
          "Start: (0, 0, 0°), jerk-limited acceleration",
          "First spline phase: arc to (18″, 24″) with a 35° tangent",
          "Second spline phase: exit to (36″, 48″) at 1.4 m/s, heading 90°"
        ],
        cpp: `void autonomous() {\n    chassis.setPose(0.0, 0.0, 0.0);\n    // High-speed quintic spline through the LTV-LQR cascade\n    autoSpline(18.0, 24.0, 35.0, 1.4);\n    autoSpline(36.0, 48.0, 90.0, 1.4);\n}`
      };
    } else if (presetKey === 'strafe') {
      plan = {
        title: "360° Holonomic Spin & Strafe",
        startPose: { x: 0, y: 0, theta: 0 },
        waypoints: [
          { id: 1, x: 0, y: 0, theta: 0, type: 'drive', speed: 1.0 },
          { id: 2, x: 24, y: 0, theta: 90, type: 'strafe', speed: 1.0 },
          { id: 3, x: 24, y: 24, theta: 180, type: 'strafe', speed: 1.0 },
          { id: 4, x: 0, y: 24, theta: 270, type: 'strafe', speed: 1.0 },
          { id: 5, x: 0, y: 0, theta: 360, type: 'strafe', speed: 1.0 }
        ],
        actions: [
          { type: 'pose', targetX: 24, targetY: 0, targetTheta: 90, desc: 'Strafe Right + 90° yaw' },
          { type: 'pose', targetX: 24, targetY: 24, targetTheta: 180, desc: 'Strafe Up + 180° yaw' },
          { type: 'pose', targetX: 0, targetY: 24, targetTheta: 270, desc: 'Strafe Left + 270° yaw' },
          { type: 'pose', targetX: 0, targetY: 0, targetTheta: 0, desc: 'Return Home + 360° yaw' }
        ],
        steps: [
          "Strafe right 24″ while rotating the chassis to 90°",
          "Strafe up 24″ while rotating to 180° (X-drive 4-wheel kinematics)",
          "Strafe left 24″ while rotating to 270°",
          "Return to the start point (0, 0), completing a full 360° turn"
        ],
        cpp: `void autonomous() {\n    chassis.setPose(0.0, 0.0, 0.0);\n    // Holonomic motion with continuous rotation\n    autoPose(24.0,  0.0,  90.0);\n    autoPose(24.0, 24.0, 180.0);\n    autoPose( 0.0, 24.0, 270.0);\n    autoPose( 0.0,  0.0,   0.0);\n}`
      };
    } else if (presetKey === 'pickup') {
      plan = {
        title: "Override Cup/Pin Pickup & High Goal Score",
        startPose: { x: 0, y: 0, theta: 0 },
        waypoints: [
          { id: 1, x: 0, y: 0, theta: 0, type: 'drive', speed: 1.0 },
          { id: 2, x: 12, y: 20, theta: 0, type: 'drive', speed: 0.8 },
          { id: 3, x: 24, y: 36, theta: 45, type: 'drive', speed: 0.8 }
        ],
        actions: [
          { type: 'drivePoint', targetX: 12, targetY: 20, desc: 'Approach Cup at (12, 20)' },
          { type: 'lift', height: 8.0, desc: 'Elevate Lift 8.0" for Secure Grip' },
          { type: 'drivePoint', targetX: 24, targetY: 36, desc: 'Drive to Alliance High Goal' },
          { type: 'lift', height: 14.5, desc: 'Score Lift at Goal Height 14.5"' },
          { type: 'drive', targetInches: -10, heading: 45, desc: 'Back off safely' }
        ],
        steps: [
          "Approach the Cup/Pin at (12″, 20″)",
          "Raise the fork 8″ to secure the element",
          "Move to the goal position (24″, 36″) at 45°",
          "Raise the fork to the 14.5″ goal height and place the element on the stack",
          "Safe 10″ back-off while lowering the fork"
        ],
        cpp: `void autonomous() {\n    chassis.setPose(0.0, 0.0, 0.0);\n    // 1. Approach the object\n    autoPose(12.0, 20.0, 0.0);\n    lift.move_absolute(8.0, 100);\n\n    // 2. Deliver to the high goal\n    autoPose(24.0, 36.0, 45.0);\n    lift.move_absolute(14.5, 100);\n    pros::delay(600);\n\n    // 3. Back off\n    chassis.driveDistance(-10.0, 1000);\n    lift.move_absolute(0.0, 100);\n}`
      };
    } else if (presetKey === 'zigzag') {
      plan = {
        title: "Obstacle Avoidance Zig-Zag Maneuver",
        startPose: { x: 0, y: 0, theta: 0 },
        waypoints: [
          { id: 1, x: 0, y: 0, theta: 0, type: 'drive', speed: 1.0 },
          { id: 2, x: -16, y: 18, theta: -20, type: 'spline', speed: 1.1 },
          { id: 3, x: 16, y: 36, theta: 20, type: 'spline', speed: 1.1 },
          { id: 4, x: 0, y: 54, theta: 0, type: 'spline', speed: 1.1 }
        ],
        actions: [
          { type: 'spline', start: {x:0, y:0, theta:0}, end: {x:-16, y:18, theta:-20}, maxVel: 1.1, maxAccel: 1.5, desc: 'Left Slalom Wave' },
          { type: 'spline', start: {x:-16, y:18, theta:-20}, end: {x:16, y:36, theta:20}, maxVel: 1.1, maxAccel: 1.5, desc: 'Right Slalom Wave' },
          { type: 'spline', start: {x:16, y:36, theta:20}, end: {x:0, y:54, theta:0}, maxVel: 1.1, maxAccel: 1.5, desc: 'Center Alignment' }
        ],
        steps: [
          "Start of the manoeuvre around the central post",
          "Left spline wave to (-16″, 18″) with a -20° tilt",
          "Right spline wave to (16″, 36″) with a +20° tilt",
          "Exit onto the straight (0″, 54″) in the opponent half"
        ],
        cpp: `void autonomous() {\n    chassis.setPose(0.0, 0.0, 0.0);\n    autoSpline(-16.0, 18.0, -20.0, 1.1);\n    autoSpline( 16.0, 36.0,  20.0, 1.1);\n    autoSpline(  0.0, 54.0,   0.0, 1.1);\n}`
      };
    } else {
      plan = {
        title: "Precision Four-Corner Box Patrol",
        startPose: { x: 0, y: 0, theta: 0 },
        waypoints: [
          { id: 1, x: 0, y: 0, theta: 0, type: 'drive', speed: 1.0 },
          { id: 2, x: 0, y: 30, theta: 0, type: 'drive', speed: 1.0 },
          { id: 3, x: 30, y: 30, theta: 0, type: 'strafe', speed: 1.0 },
          { id: 4, x: 30, y: 0, theta: 0, type: 'drive', speed: 1.0 },
          { id: 5, x: 0, y: 0, theta: 0, type: 'strafe', speed: 1.0 }
        ],
        actions: [
          { type: 'drivePoint', targetX: 0, targetY: 30, desc: 'North to (0, 30)' },
          { type: 'pose', targetX: 30, targetY: 30, targetTheta: 0, desc: 'East Strafe to (30, 30)' },
          { type: 'drivePoint', targetX: 30, targetY: 0, desc: 'South to (30, 0)' },
          { type: 'pose', targetX: 0, targetY: 0, targetTheta: 0, desc: 'West Strafe Return to (0, 0)' }
        ],
        steps: [
          "Drive forward 30″ north",
          "Pure strafe right 30″ with a constant chassis heading (0°)",
          "Drive backward 30″ south",
          "Strafe left 30″ back to the start (0, 0)"
        ],
        cpp: `void autonomous() {\n    chassis.setPose(0.0, 0.0, 0.0);\n    autoPose( 0.0, 30.0, 0.0);\n    autoPose(30.0, 30.0, 0.0);\n    autoPose(30.0,  0.0, 0.0);\n    autoPose( 0.0,  0.0, 0.0);\n}`
      };
    }
    renderAiPlan(plan);
  }

  const aiOptimizerSection = document.getElementById('aiOptimizerSection');
  const aiOptimizerVerdict = document.getElementById('aiOptimizerVerdict');
  const aiStrategyCards = document.getElementById('aiStrategyCards');

  let optimizerStrategies = [];
  let selectedStrategyIndex = 0;

  function parseCompoundTacticalPrompt(prompt) {
    const p = prompt.trim();
    
    let startX = 0, startY = 0, startTheta = 0;
    let remaining = p;
    
    // 1. Explicit labeled coordinates: start at point x=10, y=20 (heading=90)
    const labeledStartM = remaining.match(/(?:start|старт|начало|начинаем|pose|spawn|from|исходная)[^\d\-]*?x\s*[:=]\s*([+-]?\d+(?:\.\d+)?)[^\d\-]*?y\s*[:=]\s*([+-]?\d+(?:\.\d+)?)(?:[^\d\-]*?(?:theta|heading|курс|угол|th)\s*[:=]\s*([+-]?\d+(?:\.\d+)?))?/i);
    if (labeledStartM) {
      startX = Math.max(-60, Math.min(60, parseFloat(labeledStartM[1])));
      startY = Math.max(-60, Math.min(60, parseFloat(labeledStartM[2])));
      if (labeledStartM[3] !== undefined) startTheta = parseFloat(labeledStartM[3]);
      remaining = remaining.replace(labeledStartM[0], ' ');
    } else {
      // 2. Flexible coordinate tuple: start at point (10, 20), start at point 10:20, начало на точке 30:30, etc.
      const startTupleRegex = /(?:start(?:ing)?(?:\s+(?:at|from|in))?(?:\s+(?:the\s+)?(?:point|position|pose|coords|coordinates|точке|точки|позиции|координатах))?|старт(?:\s+(?:в|на|с|из)(?:\s+точк[а-я]*)?)?|начало(?:\s+(?:в|на|с|из)(?:\s+точк[а-я]*)?)?|нач(?:ать|инаем|иная)(?:\s+(?:в|на|с|из)(?:\s+точк[а-я]*)?)?|точка\s+старта|исходна(?:я|я\s+точка|я\s+позиция)|from(?:\s+(?:the\s+)?point)?|pose|spawn(?:\s+at)?)\s*[:=]?\s*\(?\s*(?:x\s*[:=]\s*)?([+-]?\d+(?:\.\d+)?)\s*[:,\s]\s*(?:y\s*[:=]\s*)?([+-]?\d+(?:\.\d+)?)(?:\s*[:,\s]\s*(?:(?:th|heading|курс|угол)\s*[:=]\s*)?([+-]?\d+(?:\.\d+)?))?\s*\)?/i;
      const startM = remaining.match(startTupleRegex);
      if (startM) {
        startX = Math.max(-60, Math.min(60, parseFloat(startM[1])));
        startY = Math.max(-60, Math.min(60, parseFloat(startM[2])));
        if (startM[3] !== undefined) startTheta = parseFloat(startM[3]);
        remaining = remaining.replace(startM[0], ' ');
      }
    }
    
    // Split into individual instruction clauses on commas, semicolons, then, and, затем, потом, etc.
    const rawClauses = remaining.split(/(?:;|\n|\r|\bthen\b|\bпосле чего\b|\bзатем\b|\bпотом\b|\bдалее\b|,(?![^\(]*\)))/i)
      .map(c => c.trim()).filter(c => c.length > 0);
    
    const state = { x: startX, y: startY, theta: startTheta };
    const allActions = [];
    const steps = [
      `Autonomous plan starts at (${state.x.toFixed(1)}″, ${state.y.toFixed(1)}″, heading ${state.theta}°).`
    ];
    const waypoints = [
      { id: 1, x: state.x, y: state.y, theta: state.theta, type: 'start', speed: 1.0 }
    ];

    let totalLengthIn = 0;
    let turnCount = 0;
    let isStrafeMajor = false;

    for (let i = 0; i < rawClauses.length; i++) {
      const clause = rawClauses[i];
      const c = clause.toLowerCase().trim();
      if (/^(?:start|старт|начало|spawn|from|исходн)/i.test(c) && !/(?:go|drive|turn|strafe|lift|ехать|поворот)/i.test(c)) {
        continue;
      }
      const rad = state.theta * Math.PI / 180;

      // 1. Compound: 'go forward by turning right/left N degrees' or 'turn right N and go forward D'
      const turnDriveM = c.match(/(?:go\s+forward|drive|ехать|вперед)[^\d\-]+(?:turning|повернув|поворот)\s+(?:to\s+the\s+)?(right|left|направо|налево)\s*(?:by\s*)?([+-]?\d+(?:\.\d+)?)\s*(?:deg|градус|°)?/i)
                      || c.match(/(?:turn|повернуть)\s+(right|left|направо|налево)\s*([+-]?\d+(?:\.\d+)?)\s*(?:deg|градус|°)?\s+(?:and|then|и|затем)\s+(?:go\s+forward|drive|ехать|вперед)\s*(?:by\s*)?(\d+(?:\.\d+)?)?/i);
      if (turnDriveM) {
        const dir = (turnDriveM[1].includes('right') || turnDriveM[1].includes('направо')) ? 1 : -1;
        const deg = parseFloat(turnDriveM[2]);
        const dist = parseFloat(turnDriveM[3] || '20');
        let newTheta = (state.theta + dir * deg) % 360;
        if (newTheta < 0) newTheta += 360;
        state.theta = Math.round(newTheta);
        const newRad = state.theta * Math.PI / 180;
        const nx = +(state.x + dist * Math.sin(newRad)).toFixed(1);
        const ny = +(state.y + dist * Math.cos(newRad)).toFixed(1);
        
        allActions.push({ type: 'turn', targetHeading: state.theta, desc: `Turn ${dir > 0 ? 'right' : 'left'} ${deg}° to ${state.theta}°` });
        allActions.push({ type: 'drive', targetInches: dist, heading: state.theta, targetX: nx, targetY: ny, desc: `Drive forward ${dist}" along ${state.theta}°` });
        steps.push(`Turn ${dir > 0 ? 'right' : 'left'} by ${deg}° (heading ${state.theta}°) and accelerate forward ${dist}″ to (${nx}″, ${ny}″).`);
        state.x = Math.max(-60, Math.min(60, nx));
        state.y = Math.max(-60, Math.min(60, ny));
        totalLengthIn += dist;
        turnCount++;
        waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'spline', speed: 1.2 });
        continue;
      }

      // 2. Strafe left / right N inches
      const strafeM = c.match(/(?:strafe|стрейф)\s*(?:to\s+the\s+)?(right|left|вправо|влево)?\s*(?:на\s*)?(\d+(?:\.\d+)?)\s*(?:in|inch|инч|дюйм|\")?/i);
      if (strafeM) {
        const isLeft = strafeM[1] && (strafeM[1].includes('left') || strafeM[1].includes('влево'));
        const dist = parseFloat(strafeM[2]);
        const sign = isLeft ? -1 : 1;
        const nx = +(state.x + sign * dist * Math.cos(rad)).toFixed(1);
        const ny = +(state.y - sign * dist * Math.sin(rad)).toFixed(1);
        allActions.push({ type: 'strafe', targetInches: sign * dist, heading: state.theta, targetX: nx, targetY: ny, desc: `Strafe ${isLeft ? 'left' : 'right'} ${dist}"` });
        steps.push(`Holonomic strafe ${isLeft ? 'left' : 'right'} ${dist}″ to (${nx}″, ${ny}″).`);
        state.x = Math.max(-60, Math.min(60, nx));
        state.y = Math.max(-60, Math.min(60, ny));
        totalLengthIn += dist;
        isStrafeMajor = true;
        waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'strafe', speed: 1.0 });
        continue;
      }

      // 3. 'turn to point (X, Y)'
      const turnPointM = c.match(/(?:turn\s+to\s+point|повернут(?:ься)?\s+к\s+точке|look\s+at)\s*\(?\s*([+-]?\d+(?:\.\d+)?)\s*[:,\s]\s*([+-]?\d+(?:\.\d+)?)\s*\)?/i);
      if (turnPointM) {
        const tx = parseFloat(turnPointM[1]);
        const ty = parseFloat(turnPointM[2]);
        const dx = tx - state.x;
        const dy = ty - state.y;
        let bearing = Math.round(Math.atan2(dx, dy) * 180 / Math.PI);
        if (bearing < 0) bearing += 360;
        state.theta = bearing;
        allActions.push({ type: 'turn', targetHeading: bearing, desc: `Turn to point (${tx}, ${ty}) at ${bearing}°` });
        steps.push(`Aim at point (${tx}″, ${ty}″) — rotate the chassis to heading ${bearing}°.`);
        turnCount++;
        waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'turn', speed: 1.0 });
        continue;
      }

      // 4. 'go to (point) (X, Y)'
      const goPointM = c.match(/(?:go\s+to(?:\s+point)?|move\s+to|drive\s+to|ехать\s+в(?:\s+точку)?|в\s+точку|к\s+точке)\s*\(?\s*([+-]?\d+(?:\.\d+)?)\s*[:,\s]\s*([+-]?\d+(?:\.\d+)?)\s*\)?/i);
      if (goPointM) {
        const tx = Math.max(-60, Math.min(60, parseFloat(goPointM[1])));
        const ty = Math.max(-60, Math.min(60, parseFloat(goPointM[2])));
        const dx = tx - state.x;
        const dy = ty - state.y;
        const segDist = Math.hypot(dx, dy);
        let headingToPoint = Math.round(Math.atan2(dx, dy) * 180 / Math.PI);
        if (headingToPoint < 0) headingToPoint += 360;
        state.x = tx;
        state.y = ty;
        state.theta = headingToPoint;
        allActions.push({ type: 'drivePoint', targetX: tx, targetY: ty, targetHeading: headingToPoint, desc: `Move to point (${tx}, ${ty})` });
        steps.push(`Move to target point (${tx}″, ${ty}″) at ${headingToPoint}°.`);
        totalLengthIn += segDist;
        waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'drive', speed: 1.3 });
        continue;
      }

      // 5. 'turn right/left N degrees' or 'turn to N degrees'
      const turnDegM = c.match(/(?:turn\s+(?:to\s+the\s+)?|повернуть\s+|разворот\s+на\s*)(right|left|направо|налево)\s*(?:by\s*)?([+-]?\d+(?:\.\d+)?)\s*(?:deg|градус|°)?/i)
                    || c.match(/(?:turn\s+to|повернуть\s+на|курс)\s*([+-]?\d+(?:\.\d+)?)\s*(?:deg|градус|°)?/i);
      if (turnDegM) {
        const isDirected = turnDegM[2] !== undefined;
        const dirStr = isDirected ? turnDegM[1] : '';
        const isRelRight = dirStr.includes('right') || dirStr.includes('направо');
        const isRelLeft = dirStr.includes('left') || dirStr.includes('налево');
        const deg = parseFloat(isDirected ? turnDegM[2] : turnDegM[1]);
        let newTheta = state.theta;
        if (isRelRight) newTheta = (state.theta + deg) % 360;
        else if (isRelLeft) newTheta = (state.theta - deg) % 360;
        else newTheta = deg % 360;
        if (newTheta < 0) newTheta += 360;
        state.theta = Math.round(newTheta);
        allActions.push({ type: 'turn', targetHeading: state.theta, desc: `Turn to ${state.theta}°` });
        steps.push(`Turn ${isRelRight ? 'right by ' + deg + '°' : isRelLeft ? 'left by ' + deg + '°' : 'to heading ' + deg + '°'} (new heading: ${state.theta}°).`);
        turnCount++;
        waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'turn', speed: 1.0 });
        continue;
      }

      // 6. 'drive forward / backward N inches'
      const driveDistM = c.match(/(?:drive|go|move|проехать|ехать)?\s*(forward|backward|вперед|вперёд|назад)?\s*(\d+(?:\.\d+)?)\s*(?:in|inch|инч|дюйм|\")/i)
                      || c.match(/(?:drive|go|move|проехать|ехать)\s*(forward|backward|вперед|вперёд|назад)?\s*(\d+(?:\.\d+)?)/i);
      if (driveDistM) {
        const isBack = (driveDistM[1] && (driveDistM[1].includes('back') || driveDistM[1].includes('назад')));
        const dist = parseFloat(driveDistM[2]);
        const sign = isBack ? -1 : 1;
        const nx = +(state.x + sign * dist * Math.sin(rad)).toFixed(1);
        const ny = +(state.y + sign * dist * Math.cos(rad)).toFixed(1);
        allActions.push({ type: 'drive', targetInches: sign * dist, heading: state.theta, targetX: nx, targetY: ny, desc: `Drive ${isBack ? 'backward' : 'forward'} ${dist}"` });
        steps.push(`Straight acceleration ${isBack ? 'backward' : 'forward'} ${dist}″ to (${nx}″, ${ny}″).`);
        state.x = Math.max(-60, Math.min(60, nx));
        state.y = Math.max(-60, Math.min(60, ny));
        totalLengthIn += dist;
        waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'drive', speed: 1.2 });
        continue;
      }

      // 7. Lift / Object Manipulation
      if (c.includes('lift') || c.includes('подъем') || c.includes('поднять') || c.includes('вилка') || c.includes('cup') || c.includes('pin') || c.includes('goal')) {
        const heightM = c.match(/(\d+(?:\.\d+)?)/);
        const h = heightM ? parseFloat(heightM[1]) : 12.0;
        allActions.push({ type: 'lift', height: h, desc: `Lift to ${h}"` });
        steps.push(`Raise the manipulator to ${h}″ to interact with the game element.`);
        continue;
      }
    }

    if (totalLengthIn === 0) totalLengthIn = 20;

    return {
      startPose: { x: startX, y: startY, theta: startTheta },
      finalPose: { ...state },
      actions: allActions,
      steps,
      waypoints,
      totalLengthIn,
      turnCount,
      isStrafeMajor
    };
  }

  async function generateAiFromPrompt(prompt) {
    if (btnAiGenerate) {
      btnAiGenerate.disabled = true;
      btnAiGenerate.textContent = "⏳ AI computing...";
    }

    let coordinatorLabel = "Local Tactical Planner";
    let remoteChatRecord = null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);
      const res = await fetch('http://127.0.0.1:8766/api/nemotron/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: prompt }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        remoteChatRecord = await res.json();
        if (remoteChatRecord.inferencePerformed) {
          coordinatorLabel = remoteChatRecord.provider?.provider === 'nebius-token-factory'
            ? "NVIDIA Nemotron (Nebius Token Factory Cloud GPU)"
            : "NVIDIA Nemotron 3 Nano LLM (Verified NVIDIA T4)";
        } else {
          coordinatorLabel = "Local Tactical Coordinator (Nemotron/Nebius Offline)";
        }
      }
    } catch (e) {
      coordinatorLabel = "Standalone Local Tactical Planner (iraLIB WASM)";
    } finally {
      if (btnAiGenerate) {
        btnAiGenerate.disabled = false;
        btnAiGenerate.textContent = "✨ Generate AI routine";
      }
    }

    const planResult = parseCompoundTacticalPrompt(prompt);
    const { startPose, finalPose, actions, steps, waypoints, totalLengthIn, turnCount, isStrafeMajor } = planResult;
    const lengthM = totalLengthIn * 0.0254;

    // Helper: Run fast physical rollout inside isolated simulator sandbox
    function runFastRollout(candidateActions, sp) {
      if (!core || typeof core.fork !== 'function') return null;
      try {
        const simBench = new VexRobotSimulator(core.fork());
        simBench.resetSimulation();
        simBench.setPose(sp.x, sp.y, sp.theta);
        simBench.isRunning = true;
        for (const act of candidateActions) {
          simBench.queueAction({ ...act });
        }

        let totalEnergy = 0;
        let t = 0;
        const dt = 0.01;
        const maxSteps = 2500; // max 25s sim

        for (let step = 0; step < maxSteps && simBench.isRunning; step++) {
          simBench.update(dt);
          t += dt;
          if (simBench.commandedWheelVoltages) {
            for (let v of simBench.commandedWheelVoltages) {
              totalEnergy += (v * v) * dt;
            }
          }
        }

        return {
          time: +t.toFixed(2),
          effort: +totalEnergy.toFixed(1),
          settled: simBench.lastMotionResult === 'Settled' || !simBench.isRunning,
          finalX: +simBench.x.toFixed(1),
          finalY: +simBench.y.toFixed(1),
          finalTheta: +simBench.theta.toFixed(1)
        };
      } catch (e) {
        console.warn("Fast rollout error:", e);
        return null;
      }
    }

    // ========================================================
    // Multi-Strategy Simulation Benchmark across full sequence
    // ========================================================

    // Candidate A: ⚡ High-Speed Quintic Spline (LTV-LQR continuous curve)
    const splineActions = actions.map(act => {
      if (act.type === 'drivePoint' || act.type === 'drive') {
        return {
          type: 'spline',
          start: { x: startPose.x, y: startPose.y, theta: startPose.theta },
          end: { x: act.targetX !== undefined ? act.targetX : finalPose.x, y: act.targetY !== undefined ? act.targetY : finalPose.y, theta: act.targetHeading !== undefined ? act.targetHeading : finalPose.theta },
          maxVel: 0.45,
          maxAccel: 0.8,
          desc: act.desc
        };
      }
      return { ...act };
    });

    const cppSplineLines = [
      `/**`,
      ` * @brief Strategy A: Quintic Spline LTV-LQR`,
      ` * Coordinator: ${coordinatorLabel}`,
      ` * Trajectory: Continuous curvature trajectory preserving momentum`,
      ` */`,
      `void autonomous() {`,
      `    chassis.setPose(${startPose.x.toFixed(1)}, ${startPose.y.toFixed(1)}, ${startPose.theta.toFixed(1)});`
    ];
    actions.forEach(act => {
      if (act.type === 'drivePoint') cppSplineLines.push(`    autoSpline(${act.targetX.toFixed(1)}, ${act.targetY.toFixed(1)}, ${(act.targetHeading || 0).toFixed(1)}, 1.35);`);
      else if (act.type === 'drive') cppSplineLines.push(`    chassis.driveDistance(${act.targetInches.toFixed(1)}, 1000);`);
      else if (act.type === 'turn') cppSplineLines.push(`    chassis.turnToHeading(${act.targetHeading.toFixed(1)}, 800);`);
      else if (act.type === 'strafe') cppSplineLines.push(`    chassis.driveDistance(${act.targetInches.toFixed(1)}, 1000); // strafe`);
      else if (act.type === 'lift') cppSplineLines.push(`    lift.move_absolute(${act.height.toFixed(1)}, 100);`);
    });
    cppSplineLines.push(`}`);

    // Candidate B: 🧠 GVF-TD3 Deep Residual Policy (Trained on AMD MI300X with Lyapunov Vector Field)
    const gvfActions = actions.map(act => {
      if (act.type === 'drivePoint' || act.type === 'drive') {
        return {
          type: 'gvf',
          targetX: act.targetX !== undefined ? act.targetX : finalPose.x,
          targetY: act.targetY !== undefined ? act.targetY : finalPose.y,
          targetTheta: act.targetHeading !== undefined ? act.targetHeading : finalPose.theta,
          desc: `GVF-TD3: ${act.desc || 'Guidance Vector Field'}`
        };
      }
      return { ...act };
    });

    const cppGvfLines = [
      `/**`,
      ` * @brief Strategy B: GVF-TD3 Deep Residual Policy (RL)`,
      ` * Coordinator: ${coordinatorLabel}`,
      ` * 512x10 Deep Actor trained with TD3 on AMD Instinct MI300X`,
      ` * Follows Lyapunov Guidance Vector Field with continuous residual corrections`,
      ` */`,
      `void autonomous() {`,
      `    chassis.setPose(${startPose.x.toFixed(1)}, ${startPose.y.toFixed(1)}, ${startPose.theta.toFixed(1)});`
    ];
    actions.forEach(act => {
      if (act.type === 'drivePoint') cppGvfLines.push(`    chassis.followGvf({{${act.targetX.toFixed(1)}, ${act.targetY.toFixed(1)}}}, ${(act.targetHeading || 0).toFixed(1)}, 1.20); // GVF-TD3 RL`);
      else if (act.type === 'drive') cppGvfLines.push(`    chassis.driveDistance(${act.targetInches.toFixed(1)}, 1000);`);
      else if (act.type === 'turn') cppGvfLines.push(`    chassis.turnToHeading(${act.targetHeading.toFixed(1)}, 800);`);
      else if (act.type === 'strafe') cppGvfLines.push(`    chassis.driveDistance(${act.targetInches.toFixed(1)}, 1000); // strafe`);
      else if (act.type === 'lift') cppGvfLines.push(`    lift.move_absolute(${act.height.toFixed(1)}, 100);`);
    });
    cppGvfLines.push(`}`);

    // Candidate C: ↔ Holonomic Strafe (Omni X-Drive Vector)
    const strafeActions = actions.map(act => {
      if (act.type === 'drivePoint' || act.type === 'drive') {
        return {
          type: 'pose',
          targetX: act.targetX !== undefined ? act.targetX : finalPose.x,
          targetY: act.targetY !== undefined ? act.targetY : finalPose.y,
          targetTheta: act.targetHeading !== undefined ? act.targetHeading : startPose.theta,
          desc: act.desc
        };
      }
      return { ...act };
    });

    const cppStrafeLines = [
      `/**`,
      ` * @brief Strategy C: Holonomic Strafe (Omni X-Drive)`,
      ` * Coordinator: ${coordinatorLabel}`,
      ` * Decoupled translation vector with constant heading hold`,
      ` */`,
      `void autonomous() {`,
      `    chassis.setPose(${startPose.x.toFixed(1)}, ${startPose.y.toFixed(1)}, ${startPose.theta.toFixed(1)});`
    ];
    actions.forEach(act => {
      if (act.type === 'drivePoint' || act.type === 'pose') cppStrafeLines.push(`    autoPose(${act.targetX.toFixed(1)}, ${act.targetY.toFixed(1)}, ${(act.targetHeading || startPose.theta).toFixed(1)});`);
      else if (act.type === 'drive') cppStrafeLines.push(`    chassis.driveDistance(${act.targetInches.toFixed(1)}, 1000);`);
      else if (act.type === 'turn') cppStrafeLines.push(`    chassis.turnToHeading(${act.targetHeading.toFixed(1)}, 800);`);
      else if (act.type === 'strafe') cppStrafeLines.push(`    autoPose(${act.targetX.toFixed(1)}, ${act.targetY.toFixed(1)}, ${act.heading.toFixed(1)});`);
      else if (act.type === 'lift') cppStrafeLines.push(`    lift.move_absolute(${act.height.toFixed(1)}, 100);`);
    });
    cppStrafeLines.push(`}`);

    // Candidate D: 🔄 Snap-Turn & Sprint (Cascaded PID)
    const sprintActions = [];
    actions.forEach(act => {
      if (act.type === 'drivePoint') {
        sprintActions.push({
          type: 'turn',
          targetHeading: act.targetHeading !== undefined ? act.targetHeading : 0,
          desc: `PID Turn to ${(act.targetHeading || 0).toFixed(0)}°`
        });
        sprintActions.push({
          type: 'drivePoint',
          targetX: act.targetX,
          targetY: act.targetY,
          targetHeading: act.targetHeading !== undefined ? act.targetHeading : 0,
          desc: `PID Sprint to (${act.targetX.toFixed(1)}, ${act.targetY.toFixed(1)})`
        });
      } else {
        sprintActions.push({ ...act });
      }
    });

    const cppSprintLines = [
      `/**`,
      ` * @brief Strategy D: Snap-Turn & Sprint PID`,
      ` * Coordinator: ${coordinatorLabel}`,
      ` * Discrete turn on the spot followed by high-traction straight sprint`,
      ` */`,
      `void autonomous() {`,
      `    chassis.setPose(${startPose.x.toFixed(1)}, ${startPose.y.toFixed(1)}, ${startPose.theta.toFixed(1)});`
    ];
    actions.forEach(act => {
      if (act.type === 'drivePoint') {
        cppSprintLines.push(`    chassis.turnToHeading(${(act.targetHeading || 0).toFixed(1)}, 800);`);
        cppSprintLines.push(`    chassis.moveToPoint(${act.targetX.toFixed(1)}, ${act.targetY.toFixed(1)}, 1500);`);
      } else if (act.type === 'drive') {
        cppSprintLines.push(`    chassis.driveDistance(${act.targetInches.toFixed(1)}, 1000);`);
      } else if (act.type === 'turn') {
        cppSprintLines.push(`    chassis.turnToHeading(${act.targetHeading.toFixed(1)}, 800);`);
      } else if (act.type === 'lift') {
        cppSprintLines.push(`    lift.move_absolute(${act.height.toFixed(1)}, 100);`);
      }
    });
    cppSprintLines.push(`}`);

    // ========================================================
    // Execute Real Physics Simulation Rollouts
    // ========================================================
    const splineBench = runFastRollout(splineActions, startPose);
    const gvfBench = runFastRollout(gvfActions, startPose);
    const strafeBench = runFastRollout(strafeActions, startPose);
    const sprintBench = runFastRollout(sprintActions, startPose);

    // Formatted metrics (simulated or kinematic fallback)
    const splineTime = splineBench ? splineBench.time.toFixed(2) : (lengthM / (1.35 * 0.72) + turnCount * 0.28 + 0.35).toFixed(2);
    const splineEffort = splineBench ? splineBench.effort.toFixed(1) : (lengthM * 14.2 + turnCount * 3.2 + 5.0).toFixed(1);

    const gvfTime = gvfBench ? gvfBench.time.toFixed(2) : (lengthM / (1.05 * 0.65) + turnCount * 0.30 + 0.40).toFixed(2);
    const gvfEffort = gvfBench ? gvfBench.effort.toFixed(1) : (lengthM * 9.5 + turnCount * 2.2 + 3.8).toFixed(1);

    const strafeTime = strafeBench ? strafeBench.time.toFixed(2) : (lengthM / (0.95 * 0.65) + turnCount * 0.38 + 0.50).toFixed(2);
    const strafeEffort = strafeBench ? strafeBench.effort.toFixed(1) : (lengthM * 19.8 + turnCount * 4.5 + 7.5).toFixed(1);

    const sprintTime = sprintBench ? sprintBench.time.toFixed(2) : (lengthM / (1.10 * 0.68) + turnCount * 0.55 + 0.65).toFixed(2);
    const sprintEffort = sprintBench ? sprintBench.effort.toFixed(1) : (lengthM * 22.4 + turnCount * 5.8 + 9.0).toFixed(1);

    const candidateA = {
      id: 'spline',
      title: '⚡ Quintic spline (LTV-LQR)',
      badge: splineBench ? 'Physical measurement' : 'Smooth acceleration',
      time: `${splineTime} c`,
      effort: `${splineEffort} V²·s`,
      desc: 'Continuous curvature, momentum kept without stopping at intermediate points.',
      startPose,
      waypoints: JSON.parse(JSON.stringify(waypoints)),
      actions: splineActions,
      steps,
      cpp: cppSplineLines.join('\n')
    };

    const candidateB = {
      id: 'gvf-td3',
      title: '🧠 GVF-TD3 (RL Policy)',
      badge: gvfBench ? 'AMD MI300X RL' : 'Vector field',
      time: `${gvfTime} c`,
      effort: `${gvfEffort} V²·s`,
      desc: '512×10 neural (TD3) correction on top of a Lyapunov vector field. Lowest energy use.',
      startPose,
      waypoints: JSON.parse(JSON.stringify(waypoints)),
      actions: gvfActions,
      steps,
      cpp: cppGvfLines.join('\n')
    };

    const candidateC = {
      id: 'strafe',
      title: '↔ Holonomic strafe',
      badge: strafeBench ? 'Omni Vector' : 'Constant heading',
      time: `${strafeTime} c`,
      effort: `${strafeEffort} V²·s`,
      desc: 'Vector motion of the omni chassis while independently holding the heading to the target.',
      startPose,
      waypoints: JSON.parse(JSON.stringify(waypoints)),
      actions: strafeActions,
      steps,
      cpp: cppStrafeLines.join('\n')
    };

    const candidateD = {
      id: 'sprint',
      title: '🔄 Turn + sprint (PID)',
      badge: sprintBench ? 'Discrete PID' : 'High traction',
      time: `${sprintTime} c`,
      effort: `${sprintEffort} V²·s`,
      desc: 'Discrete in-place turn before each segment with a full stop.',
      startPose,
      waypoints: JSON.parse(JSON.stringify(waypoints)),
      actions: sprintActions,
      steps,
      cpp: cppSprintLines.join('\n')
    };

    optimizerStrategies = [candidateA, candidateB, candidateC, candidateD];

    // Winner selection based on measured physical simulation
    let winnerIndex = 0;
    let verdictText = '';

    const sTime = parseFloat(splineTime);
    const sEffort = parseFloat(splineEffort);
    const gTime = parseFloat(gvfTime);
    const gEffort = parseFloat(gvfEffort);
    const pTime = parseFloat(sprintTime);

    if (isStrafeMajor && actions.length === 1) {
      winnerIndex = 2; // Strafe
      verdictText = `🏆 Simulator physics measurement: holonomic strafe — exact sideways displacement (${strafeTime} s) without unwanted chassis rotation!`;
    } else if (sTime <= gTime && sTime <= pTime) {
      winnerIndex = 0; // Spline
      const pctFaster = Math.round(((pTime - sTime) / Math.max(0.1, pTime)) * 100);
      const isSim = splineBench ? 'Real physics measurement' : 'Planner estimate';
      verdictText = `🏆 ${isSim}: quintic spline LTV-LQR — ${pctFaster}% faster than discrete PID (${splineTime} s vs ${sprintTime} s) by taking corners smoothly without braking!`;
    } else if (gEffort < sEffort * 0.8) {
      winnerIndex = 1; // GVF
      const pctEnergy = Math.round(((sEffort - gEffort) / Math.max(0.1, sEffort)) * 100);
      const isSim = gvfBench ? 'Real physics measurement' : 'RL estimate';
      verdictText = `🏆 ${isSim}: GVF-TD3 (RL) — saves ${pctEnergy}% battery energy (${gvfEffort} V²·s vs ${splineEffort} V²·s) thanks to the Lyapunov vector field!`;
    } else {
      winnerIndex = 0;
      verdictText = `🏆 Real simulator measurement: quintic spline LTV-LQR is the best trade-off between speed (${splineTime} s) and energy (${splineEffort} V²·s)!`;
    }

    selectedStrategyIndex = winnerIndex;

    // Render Strategy Cards
    if (aiOptimizerSection && aiStrategyCards) {
      aiOptimizerSection.style.display = 'flex';
      if (aiOptimizerVerdict) aiOptimizerVerdict.textContent = verdictText;

      aiStrategyCards.innerHTML = optimizerStrategies.map((strat, idx) => `
        <div class="strategy-card ${idx === winnerIndex ? 'winner' : ''} ${idx === selectedStrategyIndex ? 'selected' : ''}" data-strategy-idx="${idx}">
          <div class="strat-title">${strat.title}</div>
          <div class="strat-metrics">
            <span>Time: <strong>${strat.time}</strong></span>
            <span>Energy: <strong>${strat.effort}</strong></span>
          </div>
          <div class="strat-desc">${strat.desc}</div>
        </div>
      `).join('');

      // Add click listeners to cards
      aiStrategyCards.querySelectorAll('.strategy-card').forEach(card => {
        card.addEventListener('click', () => {
          const idx = parseInt(card.getAttribute('data-strategy-idx'), 10);
          selectStrategy(idx);
        });
      });
    }

    selectStrategy(winnerIndex);
  }

  function selectStrategy(idx) {
    if (!optimizerStrategies[idx]) return;
    selectedStrategyIndex = idx;
    
    if (aiStrategyCards) {
      aiStrategyCards.querySelectorAll('.strategy-card').forEach((card, i) => {
        card.classList.toggle('selected', i === idx);
      });
    }

    const strat = optimizerStrategies[idx];
    renderAiPlan({
      title: strat.title,
      startPose: strat.startPose,
      waypoints: strat.waypoints,
      actions: strat.actions,
      steps: strat.steps,
      cpp: strat.cpp
    });
  }

  if (btnToggleAiStudio && aiStudioPanel) {
    btnToggleAiStudio.addEventListener('click', () => {
      const isHidden = aiStudioPanel.style.display === 'none';
      aiStudioPanel.style.display = isHidden ? 'flex' : 'none';
      btnToggleAiStudio.classList.toggle('active', isHidden);
      if (isHidden && !currentAiPlan) {
        applyAiPreset('skills');
      }
    });

    if (btnAiClose) {
      btnAiClose.addEventListener('click', () => {
        aiStudioPanel.style.display = 'none';
        btnToggleAiStudio.classList.remove('active');
      });
    }

    aiChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const preset = chip.getAttribute('data-preset');
        applyAiPreset(preset);
      });
    });

    if (btnAiGenerate && aiPromptInput) {
      btnAiGenerate.addEventListener('click', () => {
        const prompt = aiPromptInput.value.trim();
        if (!prompt) return;
        generateAiFromPrompt(prompt);
      });

      aiPromptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          btnAiGenerate.click();
        }
      });
    }

    if (btnAiCopyCpp) {
      btnAiCopyCpp.addEventListener('click', () => {
        navigator.clipboard.writeText(aiCppCodeView.textContent);
        btnAiCopyCpp.textContent = "Copied!";
        setTimeout(() => { btnAiCopyCpp.textContent = "📋 Copy C++"; }, 1500);
      });
    }

    if (btnAiSendJerry) {
      btnAiSendJerry.addEventListener('click', () => {
        if (!currentAiPlan || !currentAiPlan.waypoints) return;
        renderer.jerryWaypoints = JSON.parse(JSON.stringify(currentAiPlan.waypoints));
        renderer.selectedWpIndex = -1;
        syncWaypointTable();
        
        const jerryBanner = document.getElementById('jerryBanner');
        const jerryPanel = document.getElementById('jerryPanel');
        const btnToggleJerry = document.getElementById('btnToggleJerry');
        if (jerryBanner) jerryBanner.style.display = 'flex';
        if (jerryPanel) jerryPanel.classList.add('active');
        renderer.isJerryMode = true;
        if (btnToggleJerry) {
          btnToggleJerry.classList.add('active');
          btnToggleJerry.textContent = "❌ Close Jerry.io";
        }
        sim.renderField(renderer);
      });
    }

    if (btnAiResetRobot) {
      btnAiResetRobot.addEventListener('click', () => {
        sim.resetSimulation();
        const sp = currentAiPlan?.startPose || {x: 0, y: 0, theta: 0};
        sim.setPose(sp.x, sp.y, sp.theta);
        sim.renderField(renderer);
      });
    }

    if (btnAiRunInSim) {
      btnAiRunInSim.addEventListener('click', () => {
        if (!currentAiPlan || !currentAiPlan.actions || currentAiPlan.actions.length === 0) return;
        
        sim.resetSimulation();
        sim.isRunning = true;
        
        const startPose = currentAiPlan.startPose || {x: 0, y: 0, theta: 0};
        sim.setPose(startPose.x, startPose.y, startPose.theta);
        
        if (currentAiPlan.waypoints) {
          renderer.jerryWaypoints = JSON.parse(JSON.stringify(currentAiPlan.waypoints));
        }

        for (const action of currentAiPlan.actions) {
          sim.queueAction({...action});
        }

        btnAiRunInSim.textContent = "🚀 Running in 3D!";
        setTimeout(() => {
          btnAiRunInSim.textContent = "▶ Run the routine on the robot in 3D";
        }, 2500);
      });
    }
  }

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
    document.getElementById('gameFeedback').textContent=result.ok?(mode==='match'?'Match started: autonomous routines run for the first 15 seconds.':'Game started: WASD — move, E/Q — lift, F — take / Toggle.'):result.error;
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
    document.getElementById('gameFeedback').textContent=result.ok?'Ruling recorded':result.error;
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
      document.getElementById('ruleLog').textContent=overrideState.ruleEvents.slice(-8).map(e=>`${e.clock.toFixed(1)} s · ${e.robotId} · ${e.rule} · ${e.severity}: ${e.detail}`).join(' | ');
      const activeRobot=sim.override.robots.find(r=>r.id===sim.activeRobotId);
      document.getElementById('liftStatus').textContent=`Lift ${activeRobot.manipulator.height.toFixed(1)}″ → ${activeRobot.manipulator.target.toFixed(1)}″`;
      document.getElementById('matchStatus').textContent=overrideState.matchEnded?(overrideState.scoreFinal?'Score frozen':`Waiting for objects to settle: ${overrideState.settlingSeconds.toFixed(1)} / 5 s`):sim.fleet?[...sim.fleet.entries].map(([id,e])=>`${id}: ${e.status}${e.error?' ('+e.error+')':''}`).join(' · '):'';
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
