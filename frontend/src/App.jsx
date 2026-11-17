
import { useEffect, useRef, useState } from "react";

import "./App.css";



const API_URL = (

  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"

).replace(/\/+$/, "");



const C = {

  bg: "#07111f",

  panel: "#0e1b2b",

  light: "#132439",

  border: "#263b52",

  cyan: "#42dce9",

  blue: "#76a8ff",

  violet: "#b9a0ff",

  green: "#4cdeb0",

  amber: "#f4c16c",

  red: "#ff8f9c",

  text: "#edf5ff",

  muted: "#94a9c1",

};




const FALLBACK_EVALUATION = {
  project: "FloodSense",
  model: "U-Net",
  dataset: "Kuro Siwo (test_GRD)",

  training_dataset: "Kuro Siwo GRD",
  training_samples: 20565,
  validation_samples: 5142,
  epochs: 5,

  input_channels: 7,
  output_channels: 1,
  best_validation_loss: 0.1815013488865476,

  evaluation_samples: 67490,
  pixel_accuracy_percent: 95.90581176364653,
  flood_iou_percent: 53.59736986305719,
  precision_percent: 72.96790033614369,
  recall_percent: 66.87632137197222,
  f1_score_percent: 69.78943703377604,

  limitations: [
    "Evaluation was performed on 67,490 Kuro Siwo test_GRD samples.",
    "Flood IoU and F1-score are more informative than pixel accuracy alone.",
    "The model has not been validated for operational flood warnings.",
  ],
};



const panel = {

  background: C.panel,

  border: `1px solid ${C.border}`,

  borderRadius: 16,

  padding: 20,

  minWidth: 0,

  boxSizing: "border-box",

};



const card = {

  background: `linear-gradient(145deg, ${C.light}, ${C.panel})`,

  border: `1px solid ${C.border}`,

  borderRadius: 13,

  padding: 17,

  minWidth: 0,

  boxSizing: "border-box",

};



const inputStyle = {

  width: "100%",

  boxSizing: "border-box",

  background: "#091625",

  color: C.text,

  border: `1px solid ${C.border}`,

  borderRadius: 9,

  padding: "12px 13px",

  fontSize: 12,

};



const primaryButton = {

  width: "100%",

  border: 0,

  borderRadius: 10,

  padding: "13px 16px",

  background: C.cyan,

  color: "#061521",

  fontWeight: 800,

  fontSize: 12,

  cursor: "pointer",

};



function percent(value) {

  if (value === null || value === undefined || value === "") return "—";

  const number = Number(value);

  return Number.isFinite(number) ? `${number.toFixed(2)}%` : "—";

}



function formatKey(value) {

  return String(value ?? "")

    .replace(/_/g, " ")

    .replace(/\b\w/g, (letter) => letter.toUpperCase());

}


function SectionLabel({ children }) {

  return (

    <div

      style={{

        color: C.cyan,

        fontSize: 10,

        fontWeight: 800,

        letterSpacing: "1.7px",

        textTransform: "uppercase",

        marginBottom: 9,

      }}

    >

      {children}

    </div>

  );

}



function Heading({ eyebrow, title, subtitle, right }) {

  return (

    <div

      style={{

        display: "flex",

        justifyContent: "space-between",

        alignItems: "flex-start",

        flexWrap: "wrap",

        gap: 12,

        marginBottom: 20,

      }}

    >

      <div style={{ minWidth: 0 }}>

        <SectionLabel>{eyebrow}</SectionLabel>

        <h2

          style={{

            color: C.text,

            fontSize: "clamp(19px, 2vw, 25px)",

            lineHeight: 1.3,

            margin: 0,

          }}

        >

          {title}

        </h2>

        {subtitle && (

          <p

            style={{

              color: C.muted,

              fontSize: 12,

              lineHeight: 1.8,

              margin: "8px 0 0",

            }}

          >

            {subtitle}

          </p>

        )}

      </div>

      {right}

    </div>

  );

}



function MetricCard({ label, value, subtitle, accent, icon }) {

  return (

    <article style={{ ...card, borderTop: `3px solid ${accent}` }}>

      <div

        style={{

          display: "flex",

          justifyContent: "space-between",

          gap: 8,

          color: C.muted,

          fontSize: 12,

        }}

      >

        <span>{label}</span>

        <span style={{ color: accent, fontSize: 18 }}>{icon}</span>

      </div>

      <div

        style={{

          color: accent,

          fontSize: "clamp(20px, 2.2vw, 28px)",

          fontWeight: 800,

          marginTop: 13,

          overflowWrap: "anywhere",

        }}

      >

        {value}

      </div>

      <p

        style={{

          color: C.muted,

          fontSize: 11,

          lineHeight: 1.6,

          marginBottom: 0,

        }}

      >

        {subtitle}

      </p>

    </article>

  );

}



function Notice({ children, warning = false }) {

  return (

    <div

      style={{

        marginTop: 16,

        padding: 13,

        borderRadius: 10,

        border: `1px solid ${warning ? "#71542c" : C.border}`,

        background: warning ? "#282317" : "#0a1726",

        color: warning ? "#e8cf9f" : C.muted,

        fontSize: 11,

        lineHeight: 1.8,

      }}

    >

      {children}

    </div>

  );

}



function ResponseValue({ value }) {

  if (value === null || value === undefined || value === "") {

    return <span>Not provided</span>;

  }



  if (typeof value === "boolean") {

    return <span>{value ? "Yes" : "No"}</span>;

  }



  if (Array.isArray(value)) {

    if (value.length === 0) return <span>None listed</span>;



    return (

      <ul

        style={{

          margin: "8px 0",

          paddingLeft: 22,

          textAlign: "left",

          lineHeight: 1.7,

        }}

      >

        {value.map((item, index) => (

          <li key={index} style={{ marginBottom: 7, paddingLeft: 3 }}>

            {typeof item === "object" && item !== null

              ? Object.entries(item)

                  .map(([key, val]) => `${formatKey(key)}: ${String(val)}`)

                  .join(" · ")

              : String(item)}

          </li>

        ))}

      </ul>

    );

  }



  if (typeof value === "object") {

    return (

      <div

        style={{

          display: "grid",

          gap: 8,

          marginTop: 8,

          paddingLeft: 12,

          borderLeft: `2px solid ${C.border}`,

        }}

      >

        {Object.entries(value).map(([key, val]) => (

          <div key={key}>

            <strong style={{ color: C.cyan }}>

              {formatKey(key)}:{" "}

            </strong>

            <ResponseValue value={val} />

          </div>

        ))}

      </div>

    );

  }



  return <span>{String(value)}</span>;

}


function ResponseResults({ data }) {

  if (!data || typeof data !== "object") return null;

  const preferredKeys = [
    "risk_level",
    "risk_score",
    "priority",
    "assessment_type",
    "input_indicators",
    "reasons",
    "data_gaps",
    "recommended_actions",
    "limitations",
    "message",
    "status",
  ];

  const entries = Object.entries(data);

  const ordered = [
    ...preferredKeys
      .filter((key) =>
        Object.prototype.hasOwnProperty.call(data, key)
      )
      .map((key) => [key, data[key]]),

    ...entries.filter(
      ([key]) => !preferredKeys.includes(key)
    ),
  ];

  return (
    <div style={{ display: "grid", gap: 10 }}>

      {ordered.map(([key, value]) => (

        <div
          key={key}
          style={{
            padding: 14,
            border: `1px solid ${C.border}`,
            borderRadius: 11,
            background: "#0a1726",
            color: C.text,
            fontSize: 12,
            lineHeight: 1.8,
            minWidth: 0,
          }}
        >

          <div
            style={{
              color: C.cyan,
              fontWeight: 800,
              marginBottom: 5,
            }}
          >
            {formatKey(key)}
          </div>

          <ResponseValue value={value} />

        </div>

      ))}

    </div>
  );
}


function App() {

  const [apiStatus, setApiStatus] = useState("Checking...");

  const [lastChecked, setLastChecked] = useState("");

  const [file, setFile] = useState(null);

  const [prediction, setPrediction] = useState(null);

  const [predicting, setPredicting] = useState(false);

  const [predictionError, setPredictionError] = useState("");

  const [evaluation, setEvaluation] = useState(FALLBACK_EVALUATION);

  const [evaluationSource, setEvaluationSource] = useState("fallback");

  const [evaluationLoading, setEvaluationLoading] = useState(true);

  const [viewMode, setViewMode] = useState("mask");

  const canvasRef = useRef(null);

  const [rainfallLevel, setRainfallLevel] = useState("unknown");

  const [populationExposure, setPopulationExposure] = useState("unknown");

  const [infrastructureExposed, setInfrastructureExposed] = useState(false);

  const [manualFloodPercentage, setManualFloodPercentage] = useState("0");

  const [responsePlan, setResponsePlan] = useState(null);

  const [responseLoading, setResponseLoading] = useState(false);

  const [responseError, setResponseError] = useState("");

  // ============================================================
  // SIDEBAR NAVIGATION
  // ============================================================

  const validSections = [
    "overview",
    "monitoring",
    "mapping",
    "evaluation",
    "risk",
    "response",
  ];

  const [activeSection, setActiveSection] = useState(() => {

    const section = window.location.hash.replace("#", "");

    return validSections.includes(section)
      ? section
      : "overview";

  });

  useEffect(() => {

    const updateActiveSection = () => {

      const section = window.location.hash.replace("#", "");

      setActiveSection(
        validSections.includes(section)
          ? section
          : "overview"
      );

    };

    window.addEventListener(
      "hashchange",
      updateActiveSection
    );

    return () => {

      window.removeEventListener(
        "hashchange",
        updateActiveSection
      );

    };

  }, []);


  useEffect(() => {

    let active = true;



    async function checkApi() {

      try {

        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) throw new Error("API unavailable");



        if (active) {

          setApiStatus("Online");

          setLastChecked(new Date().toLocaleTimeString());

        }

      } catch {

        if (active) {

          setApiStatus("Offline");

          setLastChecked(new Date().toLocaleTimeString());

        }

      }

    }



    async function loadEvaluation() {

      try {

        const response = await fetch(`${API_URL}/evaluation`);

        const data = await response.json();



        if (!response.ok) {

          throw new Error(data.detail || "Evaluation report unavailable.");

        }



        if (active) {

          setEvaluation(data);

          setEvaluationSource("backend");

        }

      } catch {

        if (active) {

          setEvaluation(FALLBACK_EVALUATION);

          setEvaluationSource("fallback");

        }

      } finally {

        if (active) setEvaluationLoading(false);

      }

    }



    checkApi();

    loadEvaluation();



    const interval = setInterval(checkApi, 15000);



    return () => {

      active = false;

      clearInterval(interval);

    };

  }, []);



  async function runPrediction(event) {

    event.preventDefault();



    if (!file) {

      setPredictionError("Please select a GeoTIFF file first.");

      return;

    }



    setPredicting(true);

    setPredictionError("");

    setPrediction(null);

    setResponsePlan(null);

    setResponseError("");



    try {

      const formData = new FormData();

      formData.append("file", file);



      const response = await fetch(`${API_URL}/predict`, {

        method: "POST",

        body: formData,

      });



      const data = await response.json().catch(() => ({}));



      if (!response.ok) {

        throw new Error(

          data.detail || data.message || "Flood prediction failed."

        );

      }



      setPrediction(data);

      setManualFloodPercentage(

        String(data.predicted_flood_pixel_percentage ?? 0)

      );

      setViewMode("mask");

    } catch (error) {

      setPredictionError(

        error.message || "Could not connect to the prediction API."

      );

    } finally {

      setPredicting(false);

    }

  }



  async function generateResponsePlan(event) {

    event.preventDefault();



    setResponseLoading(true);

    setResponseError("");

    setResponsePlan(null);



    const floodPercentage = Number(

      prediction?.predicted_flood_pixel_percentage ?? manualFloodPercentage

    );



    if (

      !Number.isFinite(floodPercentage) ||

      floodPercentage < 0 ||

      floodPercentage > 100

    ) {

      setResponseError("Flood percentage must be between 0 and 100.");

      setResponseLoading(false);

      return;

    }



    const payload = {

      predicted_flood_pixel_percentage: floodPercentage,

      rainfall_level: rainfallLevel,

      population_exposure: populationExposure,

      critical_infrastructure_exposed: infrastructureExposed,

    };



    try {

      const response = await fetch(`${API_URL}/response-plan`, {

        method: "POST",

        headers: {

          Accept: "application/json",

          "Content-Type": "application/json",

        },

        body: JSON.stringify(payload),

      });



      const data = await response.json().catch(() => ({}));



      if (!response.ok) {

        const detail =

          typeof data.detail === "string"

            ? data.detail

            : JSON.stringify(data.detail || data);



        throw new Error(

          detail || `Response plan failed (HTTP ${response.status}).`

        );

      }



      setResponsePlan(data);

    } catch (error) {

      setResponseError(

        error.message ||

          "Could not connect to the response prioritization endpoint."

      );

    } finally {

      setResponseLoading(false);

    }

  }



  const probabilityMap = prediction?.probability_map;

  const mapHeight = probabilityMap?.length || 0;

  const mapWidth = probabilityMap?.[0]?.length || 0;

  const threshold = Number(prediction?.threshold ?? 0.5);



  const floodPercentage = Number(

    prediction?.predicted_flood_pixel_percentage ?? 0

  );



  const validPixelCount = Number(

    prediction?.valid_pixel_count ??

      (prediction?.width && prediction?.height

        ? prediction.width * prediction.height

        : 0)

  );



  const predictedFloodPixelCount = Number(

    prediction?.predicted_flood_pixel_count ??

      Math.round((validPixelCount * floodPercentage) / 100)

  );



  useEffect(() => {

    const canvas = canvasRef.current;



    if (!canvas || !probabilityMap?.length || !mapWidth || !mapHeight) {

      return;

    }



    canvas.width = mapWidth;

    canvas.height = mapHeight;



    const context = canvas.getContext("2d");

    if (!context) return;



    const imageData = context.createImageData(mapWidth, mapHeight);

    const pixels = imageData.data;



    for (let y = 0; y < mapHeight; y += 1) {

      for (let x = 0; x < mapWidth; x += 1) {

        const probability = Math.max(

          0,

          Math.min(1, Number(probabilityMap[y]?.[x] ?? 0))

        );



        const index = (y * mapWidth + x) * 4;

        const isFlood = probability >= threshold;



        if (viewMode === "probability") {

          let r;

          let g;

          let b;



          if (probability < 0.5) {

            const t = probability * 2;

            r = Math.round(10 + 15 * t);

            g = Math.round(24 + 95 * t);

            b = Math.round(55 + 160 * t);

          } else {

            const t = (probability - 0.5) * 2;

            r = Math.round(25 + 220 * t);

            g = Math.round(119 + 115 * t);

            b = Math.round(215 - 170 * t);

          }



          pixels[index] = r;

          pixels[index + 1] = g;

          pixels[index + 2] = b;

          pixels[index + 3] = 255;

        } else if (isFlood) {

          pixels[index] = 30;

          pixels[index + 1] = 224;

          pixels[index + 2] = 211;

          pixels[index + 3] = 255;

        } else {

          pixels[index] = 11;

          pixels[index + 1] = 25 + Math.round(probability * 22);

          pixels[index + 2] = 43 + Math.round(probability * 35);

          pixels[index + 3] = 255;

        }

      }

    }



    context.putImageData(imageData, 0, 0);

  }, [probabilityMap, mapWidth, mapHeight, threshold, viewMode]);



  const accuracy = evaluation?.pixel_accuracy_percent;

  const floodIoU = evaluation?.flood_iou_percent;



  const smallButton = (active) => ({

    border: `1px solid ${active ? C.cyan : C.border}`,

    borderRadius: 9,

    padding: "10px 13px",

    fontSize: 11,

    fontWeight: 750,

    cursor: "pointer",

    background: active ? "#123a49" : "#101e30",

    color: active ? C.cyan : C.muted,

  });



  return (

    <div

      className="app-shell"

      style={{

        minHeight: "100vh",

        background: C.bg,

        color: C.text,

      }}

    >

      <aside

        className="sidebar"

        style={{

          background: "#0a1524",

          borderRight: `1px solid ${C.border}`,

        }}

      >

        <div className="brand">

          <div

            className="brand-icon"

            style={{

              background: `linear-gradient(135deg, ${C.cyan}, ${C.blue})`,

              color: "#071421",

              borderRadius: 13,

              fontWeight: 900,

            }}

          >

            F

          </div>

          <div>

            <h2>FloodSense</h2>

            <span>FLOOD INTELLIGENCE</span>

          </div>

        </div>



        <p className="nav-label">WORKSPACE</p>



        <nav>

          <a

            className={`nav-item ${activeSection === "overview" ? "active" : ""}`}

            href="#overview"

          >

            ▦ Overview

          </a>



          <a

            className={`nav-item ${activeSection === "monitoring" ? "active" : ""}`}

            href="#monitoring"

          >

            ◉ Flood monitoring

          </a>



          <a

            className={`nav-item ${activeSection === "mapping" ? "active" : ""}`}

            href="#mapping"

          >

            ⌖ Extent mapping

          </a>



          <a

            className={`nav-item ${activeSection === "evaluation" ? "active" : ""}`}

            href="#evaluation"

          >

            ▥ Model evaluation

          </a>



          <a

            className={`nav-item ${activeSection === "risk" ? "active" : ""}`}

            href="#risk"

          >

            △ Risk assessment

          </a>



          <a

            className={`nav-item ${activeSection === "response" ? "active" : ""}`}

            href="#response"

          >

            ⇢ Response planning

          </a>

        </nav>



        <div className="sidebar-bottom">

          <div className="system-label">SYSTEM STATUS</div>

          <div className="status-line">

            <span

              className={`status-dot ${apiStatus === "Online" ? "green" : ""}`}

            />

            API {apiStatus}

          </div>

          <p>FloodSense · Development</p>

          {lastChecked && (

            <p style={{ fontSize: 10 }}>Last check: {lastChecked}</p>

          )}

        </div>

      </aside>

      <main

        className="main-content"

        id="overview"

        style={{

          minWidth: 0,

          width: "100%",

          boxSizing: "border-box",

        }}

      >

        <header

          className="topbar"

          style={{ gap: 18, flexWrap: "wrap", paddingBottom: 20 }}

        >

          <div>

            <SectionLabel>DISASTER INTELLIGENCE PLATFORM</SectionLabel>

            <h1 style={{ margin: 0, color: C.text }}>

              Flood monitoring overview

            </h1>

            <p style={{ margin: "8px 0 0", color: C.muted, fontSize: 12 }}>

              Satellite analysis · Flood extent · Risk-based response

            </p>

          </div>



          <div

            className="environment-badge"

            style={{

              border: `1px solid ${C.border}`,

              borderRadius: 30,

              padding: "9px 13px",

            }}

          >

            <span

              className={`status-dot ${apiStatus === "Online" ? "green" : ""}`}

            />

            {apiStatus === "Online" ? "API CONNECTED" : apiStatus.toUpperCase()}

          </div>

        </header>



        <section

          className="welcome-panel"

          style={{

            position: "relative",

            overflow: "hidden",

            border: `1px solid ${C.border}`,

            borderRadius: 20,

            background:

              "radial-gradient(circle at 85% 45%, #123a51 0%, transparent 34%), linear-gradient(120deg, #10253a, #0d1a2a)",

            marginBottom: 24,

          }}

        >

          <div style={{ position: "relative", zIndex: 1, maxWidth: 680 }}>

            <SectionLabel>FLOODSENSE MONITORING CENTER</SectionLabel>

            <h2

              style={{

                color: C.text,

                fontSize: "clamp(25px, 3.3vw, 38px)",

                lineHeight: 1.2,

                margin: "12px 0",

              }}

            >

              Understand flood risk.

              <br />

              Support faster decisions.

            </h2>

            <p

              style={{

                color: C.muted,

                fontSize: 13,

                lineHeight: 1.8,

                maxWidth: 540,

              }}

            >

              A multimodal framework for flood early warning, extent mapping,

              risk assessment, and intelligent response.

            </p>

            <a

              className="primary-button"

              href="#mapping"

              style={{ textDecoration: "none" }}

            >

              Analyze satellite image <span>→</span>

            </a>

          </div>



          <div className="water-graphic" aria-hidden="true">

            <div className="orbit orbit-one" />

            <div className="orbit orbit-two" />

            <div className="water-symbol">≈</div>

            <div className="graphic-caption">

              EARTH OBSERVATION

              <br />+ FLOOD INTELLIGENCE

            </div>

          </div>

        </section>



        <section

          className="metrics-grid"

          aria-label="Platform status"

          style={{

            display: "grid",

            gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",

            gap: 14,

            marginBottom: 26,

          }}

        >

          <MetricCard

            label="API connectivity"

            value={apiStatus}

            subtitle="FastAPI health endpoint"

            accent={C.blue}

            icon="⌁"

          />

          <MetricCard

            label="Flood extent mapping"

            value={prediction ? "Prediction ready" : "Awaiting image"}

            subtitle="U-Net segmentation pipeline"

            accent={C.cyan}

            icon="▧"

          />

          <MetricCard

            label="Predicted flood share"

            value={prediction ? percent(floodPercentage) : "—"}

            subtitle="Preliminary model output"

            accent={C.violet}

            icon="△"

          />

          <MetricCard

            label="Response prioritization"

            value={responsePlan ? "Generated" : "Ready"}

            subtitle="POST /response-plan"

            accent={responsePlan ? C.green : C.amber}

            icon="⇢"

          />

        </section>



        <section

          className="content-grid"

          style={{

            display: "grid",

            gridTemplateColumns: "minmax(0, 1.8fr) minmax(270px, 0.8fr)",

            alignItems: "start",

            gap: 20,

            marginBottom: 28,

          }}

        >

          <article className="panel map-panel" id="mapping" style={panel}>

            <Heading

              eyebrow="GEOSPATIAL WORKSPACE"

              title="Flood extent prediction"

              subtitle="Upload a compatible seven-band SAR and elevation GeoTIFF to run the prediction pipeline."

              right={

                <span

                  style={{

                    padding: "7px 11px",

                    border: `1px solid ${C.border}`,

                    borderRadius: 30,

                    color: C.muted,

                    fontSize: 11,

                  }}

                >

                  {prediction ? "Prediction result" : "Ready for analysis"}

                </span>

              }

            />

            <form
              onSubmit={runPrediction}
              style={{
                display: "grid",
                gap: 16,
                padding: 20,
                border: `1px solid ${C.border}`,
                borderRadius: 16,
                background: "linear-gradient(145deg, #0d2032, #091625)",
                marginBottom: 20,
              }}
            >
              <div
                style={{
                  color: C.text,
                  fontSize: 14,
                  fontWeight: 800,
                  textAlign: "center",
                  letterSpacing: "0.3px",
                }}
              >
                Upload Satellite Imagery
              </div>

              <label
                htmlFor="satellite-file"
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();

                  const droppedFile = event.dataTransfer.files?.[0];

                  if (droppedFile) {
                    const validFile = /\.(tif|tiff)$/i.test(droppedFile.name);

                    if (validFile) {
                      setFile(droppedFile);
                      setPrediction(null);
                      setPredictionError("");
                      setResponsePlan(null);
                      setResponseError("");
                    } else {
                      setPredictionError(
                        "Please upload a valid .tif or .tiff GeoTIFF file."
                      );
                    }
                  }
                }}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 13,
                  minHeight: 185,
                  padding: "25px 18px",
                  border: `2px dashed ${file ? C.green : C.cyan}`,
                  borderRadius: 14,
                  background: file
                    ? "rgba(76, 222, 176, 0.06)"
                    : "rgba(66, 220, 233, 0.045)",
                  cursor: "pointer",
                  textAlign: "center",
                  transition: "all 0.25s ease",
                  boxSizing: "border-box",
                }}
              >
                <div
                  style={{
                    width: 62,
                    height: 62,
                    display: "grid",
                    placeItems: "center",
                    borderRadius: 18,
                    background: "rgba(66, 220, 233, 0.12)",
                    border: `1px solid ${C.border}`,
                    color: C.cyan,
                    fontSize: 30,
                  }}
                >
                  {file ? "✓" : "⇧"}
                </div>

                <div>
                  <div
                    style={{
                      color: C.text,
                      fontSize: 14,
                      fontWeight: 800,
                      marginBottom: 7,
                    }}
                  >
                    {file
                      ? "Satellite image selected"
                      : "Drag and drop your GeoTIFF here"}
                  </div>

                  <div style={{ color: C.muted, fontSize: 11 }}>
                    {file
                      ? "Your file is ready for flood analysis."
                      : "Or browse your computer to select a file"}
                  </div>
                </div>

                <span
                  style={{
                    padding: "9px 19px",
                    borderRadius: 8,
                    background: C.cyan,
                    color: "#061521",
                    fontSize: 11,
                    fontWeight: 800,
                  }}
                >
                  {file ? "Choose Another File" : "Browse Files"}
                </span>

                <div
                  style={{
                    color: C.muted,
                    fontSize: 10,
                    letterSpacing: "0.4px",
                  }}
                >
                  SUPPORTED FORMAT: 7-BAND .TIF / .TIFF
                </div>

                <input
                  id="satellite-file"
                  type="file"
                  accept=".tif,.tiff"
                  style={{ display: "none" }}
                  onChange={(event) => {
                    const selectedFile = event.target.files?.[0] || null;

                    if (
                      selectedFile &&
                      !/\.(tif|tiff)$/i.test(selectedFile.name)
                    ) {
                      setFile(null);
                      setPredictionError(
                        "Please select a valid .tif or .tiff GeoTIFF file."
                      );
                      return;
                    }

                    setFile(selectedFile);
                    setPrediction(null);
                    setPredictionError("");
                    setResponsePlan(null);
                    setResponseError("");
                  }}
                />
              </label>

              {file && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    padding: 13,
                    border: "1px solid #246b55",
                    borderRadius: 10,
                    background: "rgba(76, 222, 176, 0.06)",
                    minWidth: 0,
                  }}
                >
                  <div style={{ fontSize: 23 }}>📄</div>

                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      style={{
                        color: C.text,
                        fontSize: 11,
                        fontWeight: 750,
                        overflowWrap: "anywhere",
                      }}
                    >
                      {file.name}
                    </div>

                    <div
                      style={{
                        color: C.green,
                        fontSize: 10,
                        marginTop: 5,
                      }}
                    >
                      {(file.size / (1024 * 1024)).toFixed(2)} MB · File selected
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() => {
                      setFile(null);
                      setPrediction(null);
                      setPredictionError("");
                      setResponsePlan(null);
                      setResponseError("");

                      const input = document.getElementById("satellite-file");
                      if (input) input.value = "";
                    }}
                    style={{
                      border: "1px solid #71434b",
                      borderRadius: 8,
                      padding: "7px 10px",
                      background: "#291c28",
                      color: C.red,
                      cursor: "pointer",
                      fontSize: 10,
                      fontWeight: 700,
                    }}
                  >
                    Remove
                  </button>
                </div>
              )}

              <button
                className="primary-button"
                type="submit"
                disabled={!file || predicting || apiStatus !== "Online"}
                style={{
                  ...primaryButton,
                  opacity:
                    !file || predicting || apiStatus !== "Online" ? 0.55 : 1,
                  cursor:
                    !file || predicting || apiStatus !== "Online"
                      ? "not-allowed"
                      : "pointer",
                }}
              >
                {predicting
                  ? "Analyzing satellite imagery..."
                  : "Run Flood Prediction →"}
              </button>
            </form>

            {predictionError && (
              <Notice warning>
                <strong>Prediction error:</strong> {predictionError}
              </Notice>
            )}

            {predicting && (
              <p
                style={{
                  color: C.cyan,
                  fontSize: 12,
                  lineHeight: 1.7,
                }}
              >
                Processing the satellite image on the CPU. Please wait...
              </p>
            )}

            {prediction ? (
              <div style={{ marginTop: 18 }}>
                <div
                  style={{
                    border: `1px solid ${C.border}`,
                    borderRadius: 16,
                    padding: 16,
                    background:
                      "linear-gradient(150deg, #0b1b2b, #0a1523)",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "flex-start",
                      flexWrap: "wrap",
                      gap: 12,
                      marginBottom: 16,
                    }}
                  >
                    <div>
                      <SectionLabel>SEGMENTATION OUTPUT</SectionLabel>

                      <h3
                        style={{
                          margin: 0,
                          color: C.text,
                          fontSize: 20,
                        }}
                      >
                        {viewMode === "mask"
                          ? "Predicted flood mask"
                          : "Flood probability heatmap"}
                      </h3>
                    </div>

                    <div
                      style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: 8,
                      }}
                    >
                      <button
                        type="button"
                        onClick={() => setViewMode("mask")}
                        style={smallButton(viewMode === "mask")}
                      >
                        Flood mask
                      </button>

                      <button
                        type="button"
                        onClick={() => setViewMode("probability")}
                        style={smallButton(viewMode === "probability")}
                      >
                        Probability heatmap
                      </button>
                    </div>
                  </div>

                  {probabilityMap?.length > 0 ? (
                    <div
                      style={{
                        position: "relative",
                        overflow: "hidden",
                        borderRadius: 12,
                        border: `1px solid ${C.border}`,
                        background: "#081522",
                        width: "100%",
                        maxWidth: 620,
                        margin: "0 auto",
                      }}
                    >
                      <canvas
                        ref={canvasRef}
                        role="img"
                        aria-label={
                          viewMode === "mask"
                            ? "Predicted flood segmentation mask"
                            : "Flood probability heatmap"
                        }
                        style={{
                          display: "block",
                          width: "100%",
                          height: "auto",
                          aspectRatio: `${mapWidth} / ${mapHeight}`,
                          objectFit: "contain",
                          imageRendering: "pixelated",
                          background: "#0b1b2b",
                        }}
                      />

                      <div
                        style={{
                          position: "absolute",
                          top: 12,
                          left: 12,
                          background: "rgba(5, 15, 27, .88)",
                          border: `1px solid ${C.border}`,
                          borderRadius: 8,
                          padding: "8px 10px",
                          color: C.text,
                          fontSize: 10,
                        }}
                      >
                        {mapWidth} × {mapHeight} PX · THRESHOLD{" "}
                        {threshold.toFixed(2)}
                      </div>
                    </div>
                  ) : (
                    <Notice warning>
                      The API returned no probability map to display.
                    </Notice>
                  )}

                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 12,
                      flexWrap: "wrap",
                      color: C.muted,
                      fontSize: 11,
                      marginTop: 15,
                    }}
                  >
                    {viewMode === "mask" ? (
                      <>
                        <span>
                          <span style={{ color: C.cyan }}>■</span>{" "}
                          Predicted flood (at/above threshold)
                        </span>

                        <span>
                          <span style={{ color: "#16334f" }}>■</span>{" "}
                          Below threshold
                        </span>
                      </>
                    ) : (
                      <>
                        <span>Low probability</span>

                        <div
                          style={{
                            flex: "1 1 100px",
                            maxWidth: 220,
                            height: 8,
                            borderRadius: 20,
                            background:
                              "linear-gradient(90deg, #0a1837, #1977d7, #19e0d3, #ffeb69)",
                          }}
                        />

                        <span>High probability</span>
                      </>
                    )}
                  </div>
                </div>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "repeat(auto-fit, minmax(140px, 1fr))",
                    gap: 12,
                    marginTop: 16,
                  }}
                >
                  <MetricCard
                    label="Predicted flood pixels"
                    value={predictedFloodPixelCount.toLocaleString()}
                    subtitle="Pixel count"
                    accent={C.cyan}
                    icon="▧"
                  />

                  <MetricCard
                    label="Flood pixel share"
                    value={percent(floodPercentage)}
                    subtitle="Predicted share"
                    accent={C.violet}
                    icon="△"
                  />

                  <MetricCard
                    label="Image dimensions"
                    value={`${prediction.width ?? mapWidth} × ${
                      prediction.height ?? mapHeight
                    }`}
                    subtitle="Width × height"
                    accent={C.blue}
                    icon="⌖"
                  />
                </div>

                <Notice warning>
                  <strong>Research prototype:</strong>{" "}
                  {String(
                    prediction.warning ||
                      "Predictions are experimental and have not been validated for operational flood warnings."
                  ).replace(/^Research prototype:\s*/i, "")}
                </Notice>
              </div>
            ) : (
              !predicting && (
                <div
                  className="map-placeholder"
                  style={{
                    minHeight: 240,
                    borderRadius: 16,
                    border: `1px dashed ${C.border}`,
                    background:
                      "radial-gradient(circle at center, #122b40, #091522 70%)",
                  }}
                >
                  <div className="map-grid" />

                  <div className="map-message">
                    <span className="map-pin">⌖</span>

                    <h3>Mapping workspace ready</h3>

                    <p>
                      Upload a compatible seven-band GeoTIFF containing
                      six SAR channels and one elevation channel to
                      generate a preliminary flood segmentation.
                    </p>
                  </div>

                  <span className="map-coordinate">
                    SAR + ELEVATION DATA · WAITING FOR INPUT
                  </span>
                </div>
              )
            )}

          </article>

          <article className="panel" id="monitoring" style={panel}>

            <Heading
              eyebrow="OPERATIONAL STATUS"
              title="Monitoring services"
              subtitle="Current application component status."
            />

            {[
              {
                icon: "API",
                title: "Backend service",
                description: "FastAPI · /health",
                status: apiStatus,
                good: apiStatus === "Online",
              },
              {
                icon: "AI",
                title: "Segmentation model",
                description: "U-Net prediction endpoint",
                status: prediction
                  ? "Prediction run"
                  : "Awaiting prediction",
                good: Boolean(prediction),
              },
              {
                icon: "EO",
                title: "Earth observation data",
                description: "Uploaded seven-band GeoTIFF",
                status: file ? "Selected" : "Awaiting input",
                good: Boolean(file),
              },
              {
                icon: "RS",
                title: "Response prioritization",
                description: "Risk-based planning",
                status: responsePlan
                  ? "Generated"
                  : "Awaiting input",
                good: Boolean(responsePlan),
              },
            ].map((service) => (

              <div
                key={service.title}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "15px 0",
                  borderBottom: `1px solid ${C.border}`,
                }}
              >

                <div
                  style={{
                    width: 40,
                    height: 40,
                    flexShrink: 0,
                    display: "grid",
                    placeItems: "center",
                    borderRadius: 11,
                    border: `1px solid ${C.border}`,
                    background: "#12243a",
                    color: C.cyan,
                    fontSize: 10,
                    fontWeight: 800,
                  }}
                >
                  {service.icon}
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>

                  <div
                    style={{
                      color: C.text,
                      fontSize: 12,
                      fontWeight: 750,
                    }}
                  >
                    {service.title}
                  </div>

                  <div
                    style={{
                      color: C.muted,
                      fontSize: 11,
                      marginTop: 5,
                      lineHeight: 1.5,
                    }}
                  >
                    {service.description}
                  </div>

                </div>

                <span
                  style={{
                    color: service.good ? C.green : C.amber,
                    background: service.good
                      ? "#10352f"
                      : "#342b1c",
                    borderRadius: 20,
                    padding: "6px 9px",
                    fontSize: 9,
                    fontWeight: 800,
                    whiteSpace: "nowrap",
                  }}
                >
                  {service.status}
                </span>

              </div>

            ))}

            <div
              style={{
                marginTop: 20,
                border: `1px solid ${C.border}`,
                borderRadius: 12,
                padding: 15,
                background: "#0a1726",
              }}
            >

              <SectionLabel>SELECTED DATASET</SectionLabel>

              <div
                style={{
                  color: C.text,
                  fontSize: 14,
                  fontWeight: 750,
                }}
              >
                Kuro Siwo
              </div>

              <p
                style={{
                  color: C.muted,
                  fontSize: 11,
                  lineHeight: 1.7,
                  marginBottom: 0,
                }}
              >
                Kuro Siwo satellite imagery is used for the
                current seven-channel flood segmentation workflow.
                The input combines six SAR channels with one
                elevation channel (DEM).
              </p>

            </div>

            <div
              style={{
                marginTop: 14,
                border: `1px solid ${C.border}`,
                borderRadius: 12,
                padding: 15,
                background: "#0a1726",
              }}
            >

              <SectionLabel>MODEL INPUT CONFIGURATION</SectionLabel>

              <div
                style={{
                  color: C.text,
                  fontSize: 12,
                  fontWeight: 750,
                  marginBottom: 10,
                }}
              >
                Seven-band GeoTIFF
              </div>

              <div
                style={{
                  display: "grid",
                  gap: 8,
                  color: C.muted,
                  fontSize: 11,
                  lineHeight: 1.6,
                }}
              >
                <div>Band 1: Flood VV</div>
                <div>Band 2: Flood VH</div>
                <div>Band 3: Section 1 VV</div>
                <div>Band 4: Section 1 VH</div>
                <div>Band 5: Section 2 VV</div>
                <div>Band 6: Section 2 VH</div>
                <div>Band 7: Digital Elevation Model (DEM)</div>
              </div>

              <p
                style={{
                  color: C.muted,
                  fontSize: 10,
                  lineHeight: 1.7,
                  marginBottom: 0,
                  marginTop: 12,
                }}
              >
                The uploaded file must contain these channels
                in the expected order. A valid file extension
                alone does not guarantee compatibility.
              </p>

            </div>

            <p
              style={{
                marginTop: 16,
                color: C.muted,
                fontSize: 11,
              }}
            >
              Last API check: {lastChecked || "Checking..."}
            </p>

          </article>

        </section>

        <section
          id="evaluation"
          style={{ marginTop: 30, scrollMarginTop: 24 }}
        >
          <Heading
            eyebrow="MODEL PERFORMANCE"
            title="Evaluation dashboard"
            subtitle="Metrics are displayed only when returned by the backend evaluation report."
            right={
              <span
                style={{
                  padding: "8px 12px",
                  borderRadius: 30,
                  border: `1px solid ${C.border}`,
                  color: evaluationSource === "backend" ? C.green : C.amber,
                  fontSize: 10,
                  fontWeight: 800,
                }}
              >
                {evaluationLoading
                  ? "LOADING REPORT..."
                  : evaluationSource === "backend"
                    ? "BACKEND REPORT"
                    : "FALLBACK INFORMATION"}
              </span>
            }
          />

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
              gap: 14,
            }}
          >
            <MetricCard
              label="Pixel accuracy"
              value={percent(accuracy)}
              subtitle="Correctly classified pixels overall"
              accent={C.cyan}
              icon="◉"
            />

            <MetricCard
              label="Flood IoU"
              value={percent(floodIoU)}
              subtitle="Overlap between predicted and true flood areas"
              accent={C.violet}
              icon="⌖"
            />

            <MetricCard
              label="Training samples"
              value={evaluation?.training_samples ?? "—"}
              subtitle="Reported training sample count"
              accent={C.blue}
              icon="▤"
            />

            <MetricCard
              label="Validation samples"
              value={evaluation?.validation_samples ?? "—"}
              subtitle="Reported validation sample count"
              accent={C.amber}
              icon="✓"
            />
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(auto-fit, minmax(min(100%, 300px), 1fr))",
              gap: 16,
              marginTop: 16,
            }}
          >
            <article style={panel}>
              <SectionLabel>MODEL DETAILS</SectionLabel>

              <h3
                style={{
                  margin: "8px 0 20px",
                  fontSize: 18,
                }}
              >
                Configuration
              </h3>

              {[
                ["Architecture", evaluation?.model || "U-Net"],
                ["Dataset", evaluation?.dataset || "Kuro Siwo (test_GRD)"],
                ["Training epochs", evaluation?.epochs ?? "—"],
              ].map(([label, value]) => (
                <div
                  key={label}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 14,
                    flexWrap: "wrap",
                    borderTop: `1px solid ${C.border}`,
                    paddingTop: 13,
                    marginTop: 13,
                    fontSize: 12,
                  }}
                >
                  <span style={{ color: C.muted }}>
                    {label}
                  </span>

                  <strong style={{ color: C.text }}>
                    {value}
                  </strong>
                </div>
              ))}
            </article>

            <article
              style={{
                ...panel,
                borderColor: "#71542c",
                background:
                  "linear-gradient(145deg, #302719, #191e29)",
              }}
            >
              <SectionLabel>IMPORTANT NOTICE</SectionLabel>

              <h3
                style={{
                  color: "#ffe0a6",
                  margin: "8px 0 12px",
                }}
              >
                Model limitations
              </h3>

              <p
                style={{
                  color: "#e8d7bb",
                  fontSize: 12,
                  lineHeight: 1.8,
                }}
              >
                Evaluation metrics should come from a verified
                evaluation run. Results may not represent
                performance on unseen flood events.
              </p>

              <ResponseValue
                value={
                  evaluation?.limitations ||
                  FALLBACK_EVALUATION.limitations
                }
              />
            </article>
          </div>
        </section>

        <section id="risk" style={{ marginTop: 28, scrollMarginTop: 24 }}>

          <article style={panel}>

            <Heading

              eyebrow="RISK ANALYSIS"

              title="Risk assessment inputs"

              subtitle="Review the flood extent and exposure indicators used by the response prioritization service."

            />



            <div

              style={{

                display: "grid",

                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",

                gap: 14,

              }}

            >

              <div style={card}>

                <div style={{ color: C.muted, fontSize: 11 }}>

                  Predicted flood share

                </div>

                <div

                  style={{

                    color: C.cyan,

                    fontSize: 27,

                    fontWeight: 800,

                    marginTop: 10,

                  }}

                >

                  {prediction ? percent(floodPercentage) : "Not available"}

                </div>

                <div style={{ color: C.muted, fontSize: 11, marginTop: 7 }}>

                  {prediction

                    ? "Taken from the latest satellite prediction."

                    : "You can enter a test percentage in the response form."}

                </div>

              </div>



              <div style={card}>

                <div style={{ color: C.muted, fontSize: 11 }}>

                  Rainfall indicator

                </div>

                <div

                  style={{

                    color: C.blue,

                    fontSize: 22,

                    fontWeight: 800,

                    marginTop: 10,

                  }}

                >

                  {rainfallLevel === "unknown"

                    ? "Not provided"

                    : formatKey(rainfallLevel)}

                </div>

                <div style={{ color: C.muted, fontSize: 11, marginTop: 7 }}>

                  User-provided response-planning input

                </div>

              </div>



              <div style={card}>

                <div style={{ color: C.muted, fontSize: 11 }}>

                  Population exposure

                </div>

                <div

                  style={{

                    color: C.violet,

                    fontSize: 22,

                    fontWeight: 800,

                    marginTop: 10,

                  }}

                >

                  {populationExposure === "unknown"

                    ? "Not provided"

                    : formatKey(populationExposure)}

                </div>

                <div style={{ color: C.muted, fontSize: 11, marginTop: 7 }}>

                  User-provided response-planning input

                </div>

              </div>

            </div>



            <Notice warning>

              Flood percentage alone is not a complete risk score. Reliable

              assessment also requires trustworthy rainfall, population,

              terrain, infrastructure, and local-condition data.

            </Notice>

          </article>

        </section>

        <section
          id="response"
          style={{ marginTop: 28, scrollMarginTop: 24 }}
        >
          <article
            style={{
              ...panel,
              borderColor: responsePlan ? "#246b55" : C.border,
              background: "linear-gradient(145deg, #10243a 0%, #0e1b2b 65%)",
            }}
          >
            <Heading
              eyebrow="DECISION SUPPORT"
              title="Response prioritization"
              subtitle="Submit flood extent and exposure indicators to the FastAPI response-planning endpoint."
              right={
                <span
                  style={{
                    padding: "7px 11px",
                    border: `1px solid ${C.border}`,
                    borderRadius: 30,
                    color: responsePlan ? C.green : C.amber,
                    fontSize: 11,
                  }}
                >
                  {responsePlan ? "Plan generated" : "Awaiting assessment"}
                </span>
              }
            />

            <form
              onSubmit={generateResponsePlan}
              style={{
                display: "grid",
                gap: 18,
                padding: 18,
                border: `1px solid ${C.border}`,
                borderRadius: 13,
                background: "#091625",
              }}
            >
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                  gap: 16,
                }}
              >
                <label
                  style={{
                    display: "grid",
                    gap: 8,
                    color: C.text,
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                >
                  Predicted flood pixel share (%)
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="0.01"
                    value={
                      prediction
                        ? String(prediction.predicted_flood_pixel_percentage ?? 0)
                        : manualFloodPercentage
                    }
                    onChange={(event) =>
                      setManualFloodPercentage(event.target.value)
                    }
                    disabled={Boolean(prediction)}
                    required
                    style={{
                      ...inputStyle,
                      opacity: prediction ? 0.75 : 1,
                    }}
                  />
                </label>

                <label
                  style={{
                    display: "grid",
                    gap: 8,
                    color: C.text,
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                >
                  Rainfall level
                  <select
                    value={rainfallLevel}
                    onChange={(event) => setRainfallLevel(event.target.value)}
                    style={inputStyle}
                  >
                    <option value="unknown">Unknown</option>
                    <option value="low">Low</option>
                    <option value="moderate">Moderate</option>
                    <option value="high">High</option>
                    <option value="very_high">Very high</option>
                  </select>
                </label>

                <label
                  style={{
                    display: "grid",
                    gap: 8,
                    color: C.text,
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                >
                  Population exposure
                  <select
                    value={populationExposure}
                    onChange={(event) =>
                      setPopulationExposure(event.target.value)
                    }
                    style={inputStyle}
                  >
                    <option value="unknown">Unknown</option>
                    <option value="low">Low</option>
                    <option value="moderate">Moderate</option>
                    <option value="high">High</option>
                    <option value="very_high">Very high</option>
                  </select>
                </label>
              </div>

              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  color: C.text,
                  fontSize: 12,
                  cursor: "pointer",
                }}
              >
                <input
                  type="checkbox"
                  checked={infrastructureExposed}
                  onChange={(event) =>
                    setInfrastructureExposed(event.target.checked)
                  }
                />
                Critical infrastructure is exposed
              </label>

              {responseError && (
                <Notice warning>
                  <strong>Response planning error:</strong> {responseError}
                </Notice>
              )}

              <button
                type="submit"
                disabled={responseLoading || apiStatus !== "Online"}
                style={{
                  ...primaryButton,
                  opacity:
                    responseLoading || apiStatus !== "Online" ? 0.6 : 1,
                }}
              >
                {responseLoading
                  ? "Generating response plan..."
                  : "Generate response prioritization →"}
              </button>
            </form>

            {responsePlan && (
              <div style={{ marginTop: 22 }}>
                <SectionLabel>GENERATED RESPONSE PLAN</SectionLabel>
                <ResponseResults data={responsePlan} />

                <button
                  type="button"
                  onClick={() => {
                    const blob = new Blob(
                      [JSON.stringify(responsePlan, null, 2)],
                      { type: "application/json" }
                    );
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement("a");
                    link.href = url;
                    link.download = "floodsense-response-plan.json";
                    link.click();
                    URL.revokeObjectURL(url);
                  }}
                  style={{
                    ...primaryButton,
                    marginTop: 14,
                    background: "#18374b",
                    color: C.cyan,
                    border: `1px solid ${C.border}`,
                  }}
                >
                  Export response plan as JSON ↓
                </button>
              </div>
            )}
          </article>
        </section>

        <footer
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 12,
            padding: "25px 0 12px",
            marginTop: 28,
            borderTop: `1px solid ${C.border}`,
            color: C.muted,
            fontSize: 11,
          }}
        >
          <span>FloodSense · Multimodal Flood Intelligence</span>
          <span>
            API: {apiStatus} · Last check: {lastChecked || "—"}
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;
