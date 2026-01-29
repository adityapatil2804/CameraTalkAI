// app.js
const { useEffect, useMemo, useRef, useState } = React;

const API_ENDPOINT = "http://localhost:8000/api/answer";
const STORAGE_KEY = "cameratalk_chat_v1";

// multi-face memory store
const FACE_STORE_KEY = "cameratalk_faces_v1";
const MAX_SAMPLES_PER_PERSON = 5;

function nowTime() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/* ---------------------------
   Face DB helpers (LocalStorage)
---------------------------- */
function loadFaceDB() {
  try {
    const raw = localStorage.getItem(FACE_STORE_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function saveFaceDB(db) {
  try {
    localStorage.setItem(FACE_STORE_KEY, JSON.stringify(db));
  } catch {}
}

// db format:
// [{ name: "Aditya", descriptors: [ [..128], [..128], ... ] }, ...]
function upsertFaceSample(name, descriptorArray) {
  const db = loadFaceDB();
  const cleanName = (name || "").trim();
  if (!cleanName) return;

  const idx = db.findIndex(
    (x) => (x.name || "").toLowerCase() === cleanName.toLowerCase()
  );

  if (idx === -1) {
    db.push({ name: cleanName, descriptors: [descriptorArray] });
  } else {
    const list = Array.isArray(db[idx].descriptors) ? db[idx].descriptors : [];
    list.push(descriptorArray);
    db[idx].descriptors = list.slice(-MAX_SAMPLES_PER_PERSON);
  }

  saveFaceDB(db);
}

function clearFaceDB() {
  try {
    localStorage.removeItem(FACE_STORE_KEY);
  } catch {}
}

function getSavedNames() {
  return loadFaceDB().map((p) => p.name).filter(Boolean);
}

function buildLabeledDescriptors() {
  const db = loadFaceDB();

  return db
    .filter(
      (p) =>
        p &&
        p.name &&
        Array.isArray(p.descriptors) &&
        p.descriptors.length > 0
    )
    .map((p) => {
      const descs = p.descriptors.map((d) => new Float32Array(d));
      return new faceapi.LabeledFaceDescriptors(p.name, descs);
    });
}

/* ---------------------------
   Live detection helpers
---------------------------- */
function isFingerQuestion(q) {
  const t = (q || "").toLowerCase();
  return t.includes("finger") || t.includes("fingers");
}

function isHeadcountQuestion(q) {
  const t = (q || "").toLowerCase();
  return (
    t.includes("headcount") ||
    (t.includes("how many") &&
      (t.includes("people") || t.includes("persons") || t.includes("faces"))) ||
    (t.includes("who") &&
      (t.includes("am i") ||
        t.includes("is this") ||
        t.includes("in the frame") ||
        t.includes("here"))) ||
    t.includes("who is here") ||
    t.includes("who is in")
  );
}

function secsAgo(ts) {
  if (!ts) return null;
  const s = (Date.now() - ts) / 1000;
  return s < 10 ? s.toFixed(1) : Math.round(s).toString();
}

/* ---------------------------
   Tracking helpers (frontend)
---------------------------- */
function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function normRect(r) {
  const x = Math.min(r.x1, r.x2);
  const y = Math.min(r.y1, r.y2);
  const w = Math.abs(r.x2 - r.x1);
  const h = Math.abs(r.y2 - r.y1);
  return { x, y, w, h };
}

/* ---------------------------
   App
---------------------------- */
function CameraTalkApp() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null); // hidden canvas for snapshots
  const recognitionRef = useRef(null);
  const streamRef = useRef(null);
  const chatEndRef = useRef(null);

  // face-api refs
  const faceReadyRef = useRef(false);
  const faceIntervalRef = useRef(null);

  // live mode refs
  const liveTickIntervalRef = useRef(null);
  const fingersIntervalRef = useRef(null);
  const handsRef = useRef(null);

  // smoothing for finger count
  const fingerStableRef = useRef({
    last: null,
    sameCount: 0,
    stable: null,
    lastUpdateTs: 0,
  });

  // refs to prevent stale closures (mic callback)
  const cameraOnRef = useRef(false);
  const liveOnRef = useRef(false);
  const statusRef = useRef("Idle");

  // Tracking refs
  const trackCanvasRef = useRef(null);
  const trackIntervalRef = useRef(null);
  const trackerRef = useRef(null);
  const selectingRef = useRef(false);
  const selectionRef = useRef({ x1: 0, y1: 0, x2: 0, y2: 0 });
  const lastBoxRef = useRef(null);
  const cvReadyRef = useRef(false);

  const [cameraOn, setCameraOn] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState("");

  const [textQ, setTextQ] = useState("");
  const [isSending, setIsSending] = useState(false);

  // face memory UI
  const [savedFaceNames, setSavedFaceNames] = useState(() => getSavedNames());
  const [recognizedNames, setRecognizedNames] = useState([]);

  // Live Mode UI
  const [liveOn, setLiveOn] = useState(false);
  const [liveCards, setLiveCards] = useState({
    fingers: { enabled: false, value: null, updatedAt: null },
    people: { enabled: false, value: null, updatedAt: null },
  });
  const [liveTick, setLiveTick] = useState(0);

  // Tracking UI state
  const [trackModeOn, setTrackModeOn] = useState(false);
  const [trackStatus, setTrackStatus] = useState("OFF"); // OFF | SELECT | TRACKING

  // keep refs updated
  useEffect(() => {
    cameraOnRef.current = cameraOn;
  }, [cameraOn]);
  useEffect(() => {
    liveOnRef.current = liveOn;
  }, [liveOn]);
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const refreshSavedFacesUI = () => {
    setSavedFaceNames(getSavedNames());
  };

  const [chat, setChat] = useState(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  });

  const pushUser = (text) => {
    setChat((prev) => [...prev, { role: "user", text, t: nowTime() }]);
  };

  const pushAI = (text) => {
    setChat((prev) => [...prev, { role: "ai", text, t: nowTime() }]);
  };

  // Persist chat
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chat));
    } catch {}
  }, [chat]);

  // Auto scroll
  useEffect(() => {
    if (chatEndRef.current)
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  const cancelSpeech = () => {
    if (!window.speechSynthesis) return;
    try {
      window.speechSynthesis.cancel();
    } catch {}
  };

  const stopRecognition = () => {
    const r = recognitionRef.current;
    if (!r) return;
    try {
      r.stop();
    } catch {}
    setIsListening(false);
  };

  const stopTracking = () => {
    if (trackIntervalRef.current) {
      clearInterval(trackIntervalRef.current);
      trackIntervalRef.current = null;
    }
    if (trackerRef.current) {
      try {
        // OpenCV objects do not always need explicit destroy in JS build,
        // but we'll attempt to clean references anyway.
        trackerRef.current = null;
      } catch {}
    }
    selectingRef.current = false;
    lastBoxRef.current = null;
    setTrackStatus("OFF");
    // keep trackModeOn state as-is, user might want to reselect quickly
    redrawTrackOverlay();
  };

  const clearChatAndStopAll = () => {
    stopRecognition();
    cancelSpeech();
    stopCamera();

    setChat([]);
    setTextQ("");
    setStatus("Idle");
    setError("");
    setIsSending(false);

    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {}

    clearFaceDB();
    setRecognizedNames([]);
    setSavedFaceNames([]);

    setLiveOn(false);
    stopAllLiveLoops();
    setLiveCards({
      fingers: { enabled: false, value: null, updatedAt: null },
      people: { enabled: false, value: null, updatedAt: null },
    });

    // reset finger smoother
    fingerStableRef.current = {
      last: null,
      sameCount: 0,
      stable: null,
      lastUpdateTs: 0,
    };

    // stop tracking too
    setTrackModeOn(false);
    stopTracking();
  };

  const isVideoReady = () => {
    const v = videoRef.current;
    return !!(
      v &&
      v.srcObject &&
      v.readyState >= 2 &&
      v.videoWidth > 0 &&
      v.videoHeight > 0
    );
  };

  /* ---------------------------
     Load OpenCV.js (for tracking)
  ---------------------------- */
  useEffect(() => {
    // OpenCV.js sets window.cv when ready, sometimes via onRuntimeInitialized
    const tryBind = () => {
      if (!window.cv) return false;

      // If runtime init exists, it fires when WASM is ready
      if (window.cv && typeof window.cv.onRuntimeInitialized === "function") {
        window.cv.onRuntimeInitialized = () => {
          cvReadyRef.current = true;
        };
        // If already initialized, cv.getBuildInformation exists
        if (window.cv.getBuildInformation) cvReadyRef.current = true;
      } else {
        // Some builds are ready immediately
        cvReadyRef.current = true;
      }
      return true;
    };

    if (tryBind()) return;

    // Poll a bit until cv loads
    const t = setInterval(() => {
      if (tryBind()) clearInterval(t);
    }, 200);

    return () => clearInterval(t);
  }, []);

  /* ---------------------------
     Load face-api models once
  ---------------------------- */
  useEffect(() => {
    const loadFaceModels = async () => {
      try {
        if (!window.faceapi) {
          setError("face-api.js not loaded. Check index.html script path.");
          return;
        }

        const MODEL_URL = "/models";

        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

        faceReadyRef.current = true;
      } catch (e) {
        console.error("Face model load error:", e);
        setError("Failed to load face models. Open Console (F12) for details.");
      }
    };

    loadFaceModels();
  }, []);

  /* ---------------------------
     Speech recognition setup (once, no stale state)
  ---------------------------- */
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError("Speech recognition not supported in this browser. Use Chrome.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      const text = event.results?.[0]?.[0]?.transcript || "";
      const clean = text.trim();
      if (!clean) return;

      pushUser(clean);

      if (maybeHandleRememberFaceCommand(clean)) return;

      if (liveOnRef.current) maybeEnableLiveWidgets(clean);

      handleQuestion(clean, { source: "mic" });
    };

    recognition.onerror = (event) => {
      setError(`Speech error: ${event.error}`);
      setStatus("Idle");
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      if (statusRef.current === "Listening") setStatus("Idle");
    };

    recognitionRef.current = recognition;
  }, []);

  /* ---------------------------
     Camera controls
  ---------------------------- */
  const startCamera = async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        await new Promise((resolve) => {
          videoRef.current.onloadedmetadata = () => resolve();
        });

        await videoRef.current.play().catch(() => {});
      }

      setCameraOn(true);

      // when camera starts, ensure overlay canvas matches video size
      setTimeout(() => {
        syncTrackCanvasSize();
        redrawTrackOverlay();
      }, 150);
    } catch {
      setError("Camera permission denied or camera not available.");
      setCameraOn(false);
    }
  };

  const stopFaceLoop = () => {
    if (faceIntervalRef.current) {
      clearInterval(faceIntervalRef.current);
      faceIntervalRef.current = null;
    }
    setRecognizedNames([]);
  };

  const stopCamera = () => {
    stopAllLiveLoops();
    stopFaceLoop();

    // stop tracking
    stopTracking();
    setTrackModeOn(false);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
  };

  const captureSnapshot = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;

    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return null;

    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h);
    return canvas.toDataURL("image/jpeg", 0.9);
  };

  const speakAnswer = (text) => {
    if (!window.speechSynthesis) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    cancelSpeech();
    window.speechSynthesis.speak(u);
  };

  /* ---------------------------
     Tracking: overlay draw + selection + OpenCV tracker update
  ---------------------------- */
  const syncTrackCanvasSize = () => {
    const v = videoRef.current;
    const c = trackCanvasRef.current;
    if (!v || !c) return;
    const w = v.videoWidth;
    const h = v.videoHeight;
    if (!w || !h) return;

    // We draw in real video pixels
    c.width = w;
    c.height = h;
  };

  const redrawTrackOverlay = () => {
    const c = trackCanvasRef.current;
    const v = videoRef.current;
    if (!c || !v) return;

    const ctx = c.getContext("2d");
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, c.width, c.height);

    // If selecting, draw selection rectangle
    if (selectingRef.current) {
      const r = normRect(selectionRef.current);
      if (r.w > 2 && r.h > 2) {
        ctx.strokeStyle = "rgba(255,0,0,0.95)";
        ctx.lineWidth = 3;
        ctx.strokeRect(r.x, r.y, r.w, r.h);
        ctx.fillStyle = "rgba(255,0,0,0.10)";
        ctx.fillRect(r.x, r.y, r.w, r.h);
      }
    }

    // If tracking, draw last known box
    if (lastBoxRef.current && trackStatus === "TRACKING") {
      const b = lastBoxRef.current;
      ctx.strokeStyle = "rgba(255,0,0,0.95)";
      ctx.lineWidth = 3;
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.fillStyle = "rgba(255,0,0,0.08)";
      ctx.fillRect(b.x, b.y, b.w, b.h);
    }

    // Small label
    if (trackModeOn) {
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(10, 10, 210, 34);
      ctx.fillStyle = "white";
      ctx.font = "16px Arial";
      const t =
        trackStatus === "TRACKING"
          ? "Tracking: ON"
          : trackStatus === "SELECT"
          ? "Tracking: select object"
          : "Tracking: ready";
      ctx.fillText(t, 18, 33);
    }
  };

  const getCanvasPoint = (evt) => {
    const c = trackCanvasRef.current;
    if (!c) return { x: 0, y: 0 };

    const rect = c.getBoundingClientRect();
    const px = (evt.clientX - rect.left) / rect.width;
    const py = (evt.clientY - rect.top) / rect.height;

    // Convert CSS pixel to actual canvas pixel (which equals video pixel)
    const x = clamp(Math.round(px * c.width), 0, c.width - 1);
    const y = clamp(Math.round(py * c.height), 0, c.height - 1);
    return { x, y };
  };

  const initTracker = () => {
    if (!cvReadyRef.current || !window.cv) {
      setError("OpenCV.js not ready. Make sure you added opencv.js in index.html.");
      return null;
    }

    const cv = window.cv;

    // Prefer CSRT if available, else fallback
    try {
      if (cv.TrackerCSRT_create) return cv.TrackerCSRT_create();
    } catch {}

    try {
      if (cv.TrackerKCF_create) return cv.TrackerKCF_create();
    } catch {}

    // If neither exists, tracking will not work
    return null;
  };

  const startTrackingLoop = () => {
    if (trackIntervalRef.current) return;

    const cv = window.cv;
    if (!cv || !cvReadyRef.current) return;

    trackIntervalRef.current = setInterval(() => {
      try {
        if (!cameraOnRef.current || !isVideoReady() || !trackerRef.current) return;

        const v = videoRef.current;
        const c = canvasRef.current;
        if (!v || !c) return;

        // Draw video to hidden canvas to get ImageData
        const w = v.videoWidth;
        const h = v.videoHeight;
        if (!w || !h) return;

        c.width = w;
        c.height = h;
        const ctx = c.getContext("2d");
        ctx.drawImage(v, 0, 0, w, h);

        const imgData = ctx.getImageData(0, 0, w, h);

        // Convert to OpenCV Mat
        const mat = cv.matFromImageData(imgData);

        // Update tracker
        const rect = new cv.Rect(0, 0, 0, 0);
        const ok = trackerRef.current.update(mat, rect);

        mat.delete();

        if (!ok) {
          // lost tracking
          setTrackStatus("SELECT");
          lastBoxRef.current = null;
          redrawTrackOverlay();
          return;
        }

        // Save and draw
        lastBoxRef.current = { x: rect.x, y: rect.y, w: rect.width, h: rect.height };
        redrawTrackOverlay();
      } catch {
        // silent
      }
    }, 60); // ~16 fps
  };

  const beginSelection = () => {
    if (!cameraOnRef.current || !isVideoReady()) {
      setError("Turn on the camera first and wait until ready.");
      return;
    }
    if (!trackModeOn) return;

    syncTrackCanvasSize();
    setTrackStatus("SELECT");
    selectingRef.current = false;
    lastBoxRef.current = null;
    redrawTrackOverlay();
  };

  const onTrackMouseDown = (e) => {
    if (!trackModeOn) return;
    if (trackStatus !== "SELECT") return;

    setError("");

    const p = getCanvasPoint(e);
    selectingRef.current = true;
    selectionRef.current = { x1: p.x, y1: p.y, x2: p.x, y2: p.y };
    redrawTrackOverlay();
  };

  const onTrackMouseMove = (e) => {
    if (!trackModeOn) return;
    if (!selectingRef.current) return;

    const p = getCanvasPoint(e);
    const cur = selectionRef.current;
    selectionRef.current = { ...cur, x2: p.x, y2: p.y };
    redrawTrackOverlay();
  };

  const onTrackMouseUp = (e) => {
    if (!trackModeOn) return;
    if (!selectingRef.current) return;

    selectingRef.current = false;

    const r = normRect(selectionRef.current);
    if (r.w < 20 || r.h < 20) {
      // too small
      redrawTrackOverlay();
      return;
    }

    // Init OpenCV tracker
    const tracker = initTracker();
    if (!tracker) {
      setError("OpenCV tracker not available. Try a different OpenCV.js build.");
      setTrackStatus("OFF");
      redrawTrackOverlay();
      return;
    }

    // Build initial frame Mat for init
    try {
      const cv = window.cv;
      const v = videoRef.current;
      const c = canvasRef.current;
      const w = v.videoWidth;
      const h = v.videoHeight;

      c.width = w;
      c.height = h;
      const ctx = c.getContext("2d");
      ctx.drawImage(v, 0, 0, w, h);
      const imgData = ctx.getImageData(0, 0, w, h);
      const mat = cv.matFromImageData(imgData);

      const initRect = new cv.Rect(r.x, r.y, r.w, r.h);
      const ok = tracker.init(mat, initRect);

      mat.delete();

      if (!ok) {
        setError("Failed to start tracking. Try selecting again.");
        setTrackStatus("SELECT");
        trackerRef.current = null;
        redrawTrackOverlay();
        return;
      }

      trackerRef.current = tracker;
      lastBoxRef.current = { x: r.x, y: r.y, w: r.w, h: r.h };
      setTrackStatus("TRACKING");
      redrawTrackOverlay();
      startTrackingLoop();
    } catch {
      setError("Tracking init failed. Try selecting again.");
      setTrackStatus("SELECT");
      trackerRef.current = null;
      redrawTrackOverlay();
    }
  };

  const toggleTrackMode = () => {
    setError("");

    if (trackModeOn) {
      setTrackModeOn(false);
      stopTracking();
      return;
    }

    if (!cameraOnRef.current || !isVideoReady()) {
      setError("Turn on the camera first and wait until ready.");
      return;
    }

    if (!cvReadyRef.current || !window.cv) {
      setError("OpenCV.js not ready. Add opencv.js in index.html and refresh.");
      return;
    }

    setTrackModeOn(true);
    setTrackStatus("SELECT");
    beginSelection();
  };

  // Re-sync overlay canvas when video becomes ready
  useEffect(() => {
    if (cameraOn && isVideoReady()) {
      syncTrackCanvasSize();
      redrawTrackOverlay();
    }
  }, [cameraOn, liveTick, trackModeOn, trackStatus]);

  /* ---------------------------
     Face loop
  ---------------------------- */
  const extractNameFromText = (text) => {
    const t = (text || "").trim();

    let m = t.match(
      /(?:it'?s\s+me|i\s+am|this\s+is)\s+([a-zA-Z][a-zA-Z\s]{1,30})/i
    );
    if (m && m[1]) return m[1].trim().split(" ")[0];

    m = t.match(/^([a-zA-Z]{2,20})\s+.*remember\s+my\s+face/i);
    if (m && m[1]) return m[1].trim();

    return "";
  };

  const maybeHandleRememberFaceCommand = (text) => {
    const t = (text || "").toLowerCase();
    if (!t.includes("remember") || !t.includes("face")) return false;

    const name = extractNameFromText(text);
    rememberFaceFlow(name || null);
    return true;
  };

  const rememberFaceFlow = async (nameOrNull) => {
    setError("");

    if (!cameraOnRef.current) {
      pushAI("Turn on the camera first.");
      return;
    }
    if (!isVideoReady()) {
      pushAI("Camera is still starting. Wait 1 second and try again.");
      return;
    }
    if (!faceReadyRef.current) {
      setError("Face models are still loading. Try again in a moment.");
      return;
    }

    try {
      const video = videoRef.current;

      const detections = await faceapi
        .detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 320,
            scoreThreshold: 0.5,
          })
        )
        .withFaceLandmarks()
        .withFaceDescriptors();

      if (!detections || detections.length === 0) {
        pushAI("No face detected. Move closer and face the camera.");
        return;
      }

      const biggest = detections.reduce((best, cur) => {
        const b = cur.detection.box;
        const bb = best.detection.box;
        return b.width * b.height > bb.width * bb.height ? cur : best;
      });

      let name = (nameOrNull || "").trim();
      if (!name) {
        const typed = prompt("Enter name to save (example: Aditya):", "Aditya");
        if (!typed) return;
        name = typed.trim();
      }
      if (!name) return;

      upsertFaceSample(name, Array.from(biggest.descriptor));
      refreshSavedFacesUI();

      pushAI(`Saved face sample for ${name}. I will remember until you press Clear.`);
    } catch (e) {
      console.error("Remember face error:", e);
      pushAI(
        "Face save failed. Open Console (F12) and send me the error if it repeats."
      );
    }
  };

  const updatePeopleCard = (detections, uniqueNames) => {
    setLiveCards((prev) => {
      if (!liveOnRef.current || !prev.people.enabled) return prev;

      const total = Array.isArray(detections) ? detections.length : 0;
      const names = Array.isArray(uniqueNames) ? uniqueNames : [];
      const label =
        total === 0
          ? "0"
          : names.length > 0
          ? `${total} (${names.join(", ")})`
          : `${total}`;

      return {
        ...prev,
        people: {
          ...prev.people,
          value: label,
          updatedAt: Date.now(),
        },
      };
    });
  };

  const startFaceLoop = () => {
    stopFaceLoop();

    if (!faceReadyRef.current) return;

    faceIntervalRef.current = setInterval(async () => {
      try {
        if (!cameraOnRef.current || !isVideoReady()) {
          setRecognizedNames([]);
          return;
        }

        const labeled = buildLabeledDescriptors();
        if (!labeled || labeled.length === 0) {
          setRecognizedNames([]);
          updatePeopleCard([], []);
          return;
        }

        const matcher = new faceapi.FaceMatcher(labeled, 0.55);
        const video = videoRef.current;

        const detections = await faceapi
          .detectAllFaces(
            video,
            new faceapi.TinyFaceDetectorOptions({
              inputSize: 224,
              scoreThreshold: 0.5,
            })
          )
          .withFaceLandmarks()
          .withFaceDescriptors();

        if (!detections || detections.length === 0) {
          setRecognizedNames([]);
          updatePeopleCard([], []);
          return;
        }

        const results = detections.map((d) => matcher.findBestMatch(d.descriptor));
        const names = results
          .map((r) => (r.label === "unknown" ? "" : r.label))
          .filter(Boolean);

        const unique = Array.from(new Set(names));
        setRecognizedNames(unique);

        updatePeopleCard(detections, unique);
      } catch {
        // silent
      }
    }, 1100);
  };

  useEffect(() => {
    if (cameraOn) startFaceLoop();
    else stopFaceLoop();
  }, [cameraOn]);

  /* ---------------------------
     Live Mode: cards and loops
  ---------------------------- */
  const startLiveTick = () => {
    if (liveTickIntervalRef.current) return;
    liveTickIntervalRef.current = setInterval(() => setLiveTick((x) => x + 1), 200);
  };

  const stopLiveTick = () => {
    if (liveTickIntervalRef.current) {
      clearInterval(liveTickIntervalRef.current);
      liveTickIntervalRef.current = null;
    }
  };

  const stopFingersLoop = () => {
    if (fingersIntervalRef.current) {
      clearInterval(fingersIntervalRef.current);
      fingersIntervalRef.current = null;
    }
    if (handsRef.current) {
      try {
        handsRef.current.close();
      } catch {}
      handsRef.current = null;
    }
  };

  const stopAllLiveLoops = () => {
    stopLiveTick();
    stopFingersLoop();
  };

  const maybeEnableLiveWidgets = (question) => {
    if (!liveOnRef.current) return;

    const wantsFingers = isFingerQuestion(question);
    const wantsPeople = isHeadcountQuestion(question);

    if (!wantsFingers && !wantsPeople) return;

    setLiveCards((prev) => {
      const next = { ...prev };
      if (wantsFingers) next.fingers = { ...next.fingers, enabled: true };
      if (wantsPeople) next.people = { ...next.people, enabled: true };
      return next;
    });

    startLiveTick();

    if (wantsFingers) startFingersLoop();
  };

  /* ---------------------------
     Finger counting improvements
  ---------------------------- */
  const smoothFingerCount = (raw) => {
    const s = fingerStableRef.current;

    if (s.last === raw) s.sameCount += 1;
    else {
      s.last = raw;
      s.sameCount = 1;
    }

    if (s.sameCount >= 3) s.stable = raw;

    return s.stable != null ? s.stable : raw;
  };

  const isPalmFacing = (lm) => {
    const w = lm[0];
    const i = lm[5];
    const p = lm[17];

    const v1 = { x: i.x - w.x, y: i.y - w.y, z: (i.z || 0) - (w.z || 0) };
    const v2 = { x: p.x - w.x, y: p.y - w.y, z: (p.z || 0) - (w.z || 0) };

    const crossZ = v1.x * v2.y - v1.y * v2.x;
    return crossZ;
  };

  const countFingersFromResults = (results) => {
    const multi = results && results.multiHandLandmarks;
    if (!multi || multi.length === 0) return 0;

    const handedList = results.multiHandedness || [];

    let total = 0;

    for (let idx = 0; idx < multi.length; idx++) {
      const lm = multi[idx];

      const handedLabel =
        handedList[idx]?.label ||
        handedList[idx]?.classification?.[0]?.label ||
        "Right";

      const palmSignal = isPalmFacing(lm);

      total += countOneHand(lm, handedLabel, palmSignal);
    }

    if (total < 0) total = 0;
    if (total > 10) total = 10;

    return total;
  };

  const countOneHand = (lm, handedLabel, palmSignal) => {
    const isUp = (tip, pip) => lm[tip].y < lm[pip].y;

    let count = 0;
    if (isUp(8, 6)) count++;
    if (isUp(12, 10)) count++;
    if (isUp(16, 14)) count++;
    if (isUp(20, 18)) count++;

    const thumbTipX = lm[4].x;
    const thumbIpX = lm[3].x;

    let thumbExtended =
      handedLabel === "Right" ? thumbTipX < thumbIpX : thumbTipX > thumbIpX;

    if (palmSignal < 0) thumbExtended = !thumbExtended;

    if (thumbExtended) count++;

    if (count < 0) count = 0;
    if (count > 5) count = 5;

    return count;
  };

  const ensureHands = async () => {
    if (!window.Hands) {
      throw new Error(
        "MediaPipe Hands not found. Add the hands.js script in index.html."
      );
    }
    if (handsRef.current) return handsRef.current;

    const hands = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 0,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });

    hands.onResults((results) => {
      if (!liveOnRef.current) return;

      const raw = countFingersFromResults(results);
      const stable = smoothFingerCount(raw);

      setLiveCards((prev) => {
        if (!prev.fingers.enabled) return prev;

        const now = Date.now();
        const s = fingerStableRef.current;
        const allowUpdate = now - (s.lastUpdateTs || 0) > 120;
        if (allowUpdate) s.lastUpdateTs = now;

        return {
          ...prev,
          fingers: {
            ...prev.fingers,
            value: stable == null ? "..." : String(stable),
            updatedAt: allowUpdate ? now : prev.fingers.updatedAt,
          },
        };
      });
    });

    handsRef.current = hands;
    return hands;
  };

  const startFingersLoop = async () => {
    if (!liveOnRef.current) return;
    if (fingersIntervalRef.current) return;

    try {
      await ensureHands();
    } catch (e) {
      setError(e.message || "Failed to start finger tracking.");
      return;
    }

    fingersIntervalRef.current = setInterval(async () => {
      try {
        if (!liveOnRef.current || !cameraOnRef.current || !isVideoReady()) return;
        const video = videoRef.current;
        const hands = handsRef.current;
        if (!video || !hands) return;

        await hands.send({ image: video });
      } catch {
        // silent
      }
    }, 220);
  };

  const toggleLiveMode = () => {
    setError("");

    if (liveOnRef.current) {
      setLiveOn(false);
      stopAllLiveLoops();
      return;
    }

    if (!cameraOnRef.current) {
      setError("Turn on the camera first to use Live Mode.");
      return;
    }
    if (!isVideoReady()) {
      setError("Camera is still starting. Wait 1 second, then enable Live Mode.");
      return;
    }

    setLiveOn(true);
    startLiveTick();

    setTimeout(() => {
      startFingersLoop();
    }, 0);
  };

  /* ---------------------------
     Backend question flow
  ---------------------------- */
  const handleQuestion = async (text, { source }) => {
    if (!text) return;

    if (!cameraOnRef.current) {
      pushAI("Turn on the camera first.");
      return;
    }

    if (!isVideoReady()) {
      pushAI("Camera is still starting. Wait 1 second and ask again.");
      return;
    }

    const snapshot = captureSnapshot();
    if (!snapshot) {
      pushAI("I could not capture a frame. Try again.");
      return;
    }

    setStatus("Answering");
    setError("");
    setIsSending(true);

    try {
      const res = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, image: snapshot }),
      });

      if (!res.ok) {
        const msg = await res.text().catch(() => "");
        throw new Error(msg || "Request failed");
      }

      const data = await res.json();
      const a = (data.answer || "").trim() || "No answer received.";
      pushAI(a);

      if (source === "mic") speakAnswer(a);
    } catch {
      pushAI("Backend not reachable. Is FastAPI running on port 8000?");
    } finally {
      setIsSending(false);
      setStatus("Idle");
    }
  };

  const handleMicDown = () => {
    if (!cameraOnRef.current) {
      setError("Turn on the camera first.");
      return;
    }
    if (!isVideoReady()) {
      setError("Camera is still starting. Wait 1 second.");
      return;
    }
    if (!recognitionRef.current) return;

    setError("");
    setStatus("Listening");
    setIsListening(true);

    try {
      recognitionRef.current.start();
    } catch {}
  };

  const handleMicUp = () => {
    if (!recognitionRef.current) return;
    try {
      recognitionRef.current.stop();
    } catch {}

    setIsListening(false);
    if (statusRef.current === "Listening") setStatus("Processing");
  };

  const sendTyped = async () => {
    const clean = textQ.trim();
    if (!clean) return;

    if (!cameraOnRef.current) {
      setError("Turn on the camera first.");
      return;
    }
    if (!isVideoReady()) {
      setError("Camera is still starting. Wait 1 second.");
      return;
    }
    if (isSending || isListening) return;

    setError("");
    setTextQ("");
    pushUser(clean);

    if (maybeHandleRememberFaceCommand(clean)) return;

    if (liveOnRef.current) maybeEnableLiveWidgets(clean);

    await handleQuestion(clean, { source: "text" });
  };

  const onTypedKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendTyped();
    }
  };

  const statusLabel = useMemo(() => {
    if (!cameraOn) return "Camera Off";
    if (!isVideoReady()) return "Starting";
    if (status === "Listening") return "Listening";
    if (status === "Processing") return "Processing";
    if (status === "Answering") return "Answering";
    return "Idle";
  }, [cameraOn, status]);

  const statusClass = useMemo(() => {
    if (!cameraOn) return "off";
    if (!isVideoReady()) return "starting";
    if (status === "Listening") return "listening";
    if (status === "Processing") return "processing";
    if (status === "Answering") return "answering";
    return "idle";
  }, [cameraOn, status]);

  const statusPill = useMemo(() => {
    if (!cameraOn) return "Camera: Off";
    if (status === "Listening") return "Mic: Listening";
    if (status === "Answering") return "AI: Answering";
    if (status === "Processing") return "Processing";
    if (isVideoReady()) return "Ready";
    return "Camera: Starting";
  }, [cameraOn, status]);

  const micDisabled = !cameraOn || !isVideoReady() || isSending;
  const sendDisabled = !cameraOn || !isVideoReady() || isSending || isListening;

  const onRememberFaceClick = async () => {
    if (!cameraOnRef.current || !isVideoReady()) {
      setError("Turn on camera and wait until ready.");
      return;
    }
    rememberFaceFlow(null);
  };

  const liveEnabledCards = useMemo(() => {
    const list = [];
    if (liveCards.fingers.enabled)
      list.push({ key: "fingers", title: "Fingers", data: liveCards.fingers });
    if (liveCards.people.enabled)
      list.push({ key: "people", title: "People", data: liveCards.people });
    return list;
  }, [liveCards, liveTick]);

  const liveButtonLabel = liveOn ? "Live Mode: ON" : "Live Mode: OFF";
  const trackButtonLabel = trackModeOn ? "Track Mode: ON" : "Track Mode: OFF";

  return (
    <div className="container">
      <div className="topbar">
        <div className="brand">
          <h1>CameraTalkAI</h1>
          <p>Local vision AI with Ollama. Chat is saved only in your browser.</p>
        </div>
        <div className="pills">
          <div className="pill">{statusPill}</div>
          <div className="pill">API: {API_ENDPOINT}</div>
        </div>
      </div>

      <div className="main">
        <div className="videoWrap">
          <div className="videoCard">
            <video ref={videoRef} autoPlay playsInline muted />
            {!cameraOn && <div className="overlay">Camera is off</div>}
            {cameraOn && !isVideoReady() && (
              <div className="overlay">Starting camera...</div>
            )}

            {/* Hidden canvas used for snapshots and OpenCV frame pulls */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Tracking overlay canvas (draw and select ROI here) */}
            <canvas
              ref={trackCanvasRef}
              className="trackCanvas"
              onMouseDown={onTrackMouseDown}
              onMouseMove={onTrackMouseMove}
              onMouseUp={onTrackMouseUp}
              onMouseLeave={() => {
                if (selectingRef.current) {
                  selectingRef.current = false;
                  redrawTrackOverlay();
                }
              }}
            />

            {recognizedNames.length > 0 && (
              <div className="faceTag">Recognized: {recognizedNames.join(", ")}</div>
            )}
          </div>

          <div className="controlsRow">
            <button
              className={"btn " + (cameraOn ? "danger" : "primary")}
              onClick={cameraOn ? stopCamera : startCamera}
              disabled={isListening || isSending}
            >
              {cameraOn ? "Turn Camera Off" : "Turn Camera On"}
            </button>

            <button
              className={"btn mic " + (isListening ? "active" : "")}
              onMouseDown={handleMicDown}
              onMouseUp={handleMicUp}
              onMouseLeave={handleMicUp}
              onTouchStart={handleMicDown}
              onTouchEnd={handleMicUp}
              disabled={micDisabled}
              title={micDisabled ? "Turn on camera and wait until ready" : "Hold to talk"}
            >
              {isListening ? "Listening..." : "Hold to Talk"}
            </button>

            <button
              className="btn"
              onClick={onRememberFaceClick}
              disabled={!cameraOn || !isVideoReady()}
            >
              Remember Face
            </button>

            <button
              className={"btn " + (liveOn ? "primary" : "")}
              onClick={toggleLiveMode}
              disabled={!cameraOn || !isVideoReady()}
              title="Live mode updates cards only (no chat spam)"
            >
              {liveButtonLabel}
            </button>

            <button
              className={"btn " + (trackModeOn ? "primary" : "")}
              onClick={toggleTrackMode}
              disabled={!cameraOn || !isVideoReady()}
              title="Track mode: click ON, then drag a box over the object"
            >
              {trackButtonLabel}
            </button>

            {trackModeOn && (
              <button
                className="btn"
                onClick={() => {
                  stopTracking();
                  setTrackModeOn(true);
                  setTrackStatus("SELECT");
                  beginSelection();
                }}
                disabled={!cameraOn || !isVideoReady()}
                title="Pick a new object to track"
              >
                Re-select
              </button>
            )}
          </div>

          {/* Live Cards Panel (bottom-left area) */}
          {liveEnabledCards.length > 0 && (
            <div className="livePanel">
              {liveEnabledCards.map((c) => {
                const ago = secsAgo(c.data.updatedAt);
                const showVal = c.data.value == null ? "..." : c.data.value;

                return (
                  <div key={c.key} className={"liveCard " + (liveOn ? "on" : "off")}>
                    <div className="liveTitle">{c.title}</div>
                    <div className="liveValue">{showVal}</div>
                    <div className="liveMeta">
                      {liveOn ? (ago ? `Updated ${ago}s ago` : "Waiting...") : "Live paused"}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {trackModeOn && (
            <div className="small" style={{ marginTop: "8px" }}>
              Track Mode is ON. Drag a box on the video to lock onto the object. Status: {trackStatus}
            </div>
          )}
        </div>

        <div className="chatWrap">
          <div className="chatHeader">
            <div className="chatHeaderLeft">
              <h2>Chat</h2>
              <div className={"statusBadge " + statusClass}>{statusLabel}</div>
            </div>

            <button className="btn" onClick={clearChatAndStopAll}>
              Clear
            </button>
          </div>

          <div className="chatList">
            {chat.length === 0 && (
              <div className="small">Hold the mic and ask: "What do you see?"</div>
            )}

            {chat.map((m, i) => (
              <div key={i} className={"bubble " + (m.role === "user" ? "user" : "ai")}>
                <div>{m.text}</div>
                <div className="meta">{m.t}</div>
              </div>
            ))}
            <div ref={chatEndRef}></div>
          </div>

          <div className="typeRow">
            <input
              className="typeInput"
              placeholder='Type a question (or "This is me Aditya, remember my face")'
              value={textQ}
              onChange={(e) => setTextQ(e.target.value)}
              onKeyDown={onTypedKeyDown}
              disabled={!cameraOn || !isVideoReady() || isListening || isSending}
            />
            <button className="btn" onClick={sendTyped} disabled={sendDisabled}>
              Send
            </button>
          </div>

          {error && <div className="error">{error}</div>}

          {savedFaceNames.length > 0 && (
            <div className="small" style={{ marginTop: "8px" }}>
              Saved faces: {savedFaceNames.join(", ")} (clears when you press Clear)
            </div>
          )}

          {liveOn && (
            <div className="small" style={{ marginTop: "8px" }}>
              Live Mode is ON. Ask "how many fingers" or "how many people" to show live cards.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<CameraTalkApp />);
