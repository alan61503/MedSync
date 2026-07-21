"use client";
import React, { useEffect, useState } from "react";
import UploadDropzone from "../../../components/UploadDropzone";

export default function PatientPage() {
  const [patient, setPatient] = useState<any>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [inference, setInference] = useState<any>(null);
  const [loadingInference, setLoadingInference] = useState(false);
  const [xaiMode, setXaiMode] = useState<"overlay" | "heatmap" | "original">("overlay");

  const id = typeof window !== "undefined" ? window.location.pathname.split("/").pop() : null;

  const loadPatient = async () => {
    if (!id) return;
    try {
      const res = await fetch(`http://localhost:8000/api/patients/${id}`);
      const data = await res.json();
      setPatient(data);
      if (data.images && data.images.length > 0 && !selectedImage) {
        const firstImg = data.images[0].file_path;
        setSelectedImage(firstImg);
        fetchInference(firstImg);
      }
    } catch (err) {
      console.error("Failed to load patient:", err);
    }
  };

  useEffect(() => {
    loadPatient();
  }, [id]);

  const fetchInference = async (imagePath: string) => {
    setLoadingInference(true);
    setInference(null);
    try {
      const fullUrl = imagePath.startsWith("http") ? imagePath : `http://localhost:8000${imagePath}`;
      const res = await fetch(`http://localhost:8000/api/run-inference`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_url: fullUrl }),
      });
      const data = await res.json();
      setInference(data);
    } catch (err) {
      console.error("Inference error:", err);
    } finally {
      setLoadingInference(false);
    }
  };

  if (!patient) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-6">
        <div className="flex items-center gap-3 text-sky-400">
          <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
          </svg>
          <span className="font-semibold text-lg">Loading Patient Diagnostic Record...</span>
        </div>
      </div>
    );
  }

  const osteo = inference?.osteoporosis || {};
  const riskColorClass =
    osteo.risk_color === "red" || (osteo.score >= 0.65)
      ? "bg-rose-500 text-white"
      : osteo.risk_color === "amber" || (osteo.score >= 0.38)
      ? "bg-amber-500 text-slate-950"
      : "bg-emerald-500 text-slate-950";

  return (
    <div className="space-y-6 font-sans">

        
        {/* Header Navigation & Patient Profile */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <a href="/" className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition border border-slate-700 flex items-center gap-1">
              ← Dashboard
            </a>
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-bold text-white tracking-tight">{patient.name}</h1>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-sky-500/20 text-sky-300 border border-sky-500/30">
                  ID: {patient.id}
                </span>
              </div>
              <p className="text-sm text-slate-400 mt-1">
                Age: <span className="text-slate-200 font-medium">{patient.age}</span> &bull; Gender: <span className="text-slate-200 font-medium">{patient.gender}</span> &bull; Primary Study: <span className="text-sky-400 font-semibold">Osteoporosis X-Ray & XAI Analysis</span>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 bg-slate-800/80 px-4 py-2 rounded-xl border border-slate-700/60">
            <div className="text-right">
              <div className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Diagnostic Engine</div>
              <div className="text-xs text-emerald-400 font-medium flex items-center gap-1.5 justify-end">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                DenseNet-121 + Grad-CAM XAI
              </div>
            </div>
          </div>
        </div>

        {/* Top Grid: Image Upload & Gallery */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Upload Dropzone */}
          <div className="lg:col-span-1">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-2">Upload Test X-Ray</h2>
            <UploadDropzone patientId={patient.id} onUploadComplete={loadPatient} />
          </div>

          {/* Uploaded X-Ray Selection Gallery */}
          <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col justify-between">
            <div>
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-3">
                Uploaded X-Ray Test Scans ({patient.images?.length || 0})
              </h2>

              {!patient.images || patient.images.length === 0 ? (
                <div className="p-8 text-center text-slate-500 border border-dashed border-slate-800 rounded-lg">
                  No X-ray pictures uploaded yet. Use the upload panel to add scans.
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 max-h-48 overflow-y-auto pr-1">
                  {patient.images.map((img: any) => {
                    const isSelected = selectedImage === img.file_path;
                    const imgUrl = img.file_path.startsWith("http")
                      ? img.file_path
                      : `http://localhost:8000${img.file_path}`;
                    return (
                      <div
                        key={img.id}
                        onClick={() => {
                          setSelectedImage(img.file_path);
                          fetchInference(img.file_path);
                        }}
                        className={`relative cursor-pointer rounded-lg p-2 border transition ${
                          isSelected
                            ? "bg-sky-950/60 border-sky-500 shadow-md shadow-sky-500/10"
                            : "bg-slate-800/60 border-slate-700/80 hover:border-slate-600"
                        }`}
                      >
                        <div className="h-20 bg-black/40 rounded flex items-center justify-center overflow-hidden">
                          <img src={imgUrl} alt={img.filename} className="object-contain h-full w-full" />
                        </div>
                        <div className="mt-1.5 text-xs font-medium text-slate-300 truncate">{img.filename}</div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {selectedImage && (
              <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between">
                <span className="text-xs text-slate-400 truncate">Selected: <strong className="text-slate-200">{selectedImage.split("/").pop()}</strong></span>
                <button
                  onClick={() => fetchInference(selectedImage)}
                  disabled={loadingInference}
                  className="px-4 py-1.5 bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold rounded-lg transition shadow flex items-center gap-2"
                >
                  {loadingInference ? "Analyzing XAI..." : "Re-Run Osteoporosis Inference & XAI"}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Main Section: Osteoporosis Model Results & XAI Explainable AI */}
        {loadingInference ? (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center shadow-2xl">
            <div className="inline-flex items-center gap-3 px-5 py-3 rounded-full bg-slate-800 text-sky-400 font-semibold text-sm border border-slate-700 shadow-inner">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
              </svg>
              Executing Neural Model & Grad-CAM Explainable AI (XAI) Matrix...
            </div>
          </div>
        ) : inference ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* Left Column: Osteoporosis Primary Diagnosis & Metrics (5 cols) */}
            <div className="lg:col-span-5 space-y-6">
              
              {/* Primary Osteoporosis Risk Card */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-5">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-sky-400">Primary Target Disease</h3>
                    <h2 className="text-xl font-extrabold text-white">Osteoporosis Risk Analysis</h2>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-extrabold uppercase tracking-wider ${riskColorClass}`}>
                    {osteo.risk_level || "Analyzed"}
                  </span>
                </div>

                {/* Risk Score Circle / Big Display */}
                <div className="flex items-center justify-between bg-slate-950 p-4 rounded-xl border border-slate-800">
                  <div>
                    <div className="text-xs text-slate-400 font-medium">Osteoporosis Risk Index</div>
                    <div className="text-4xl font-black text-white tracking-tight mt-1">
                      {osteo.percentage ?? ((osteo.score ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="w-24 text-right">
                    <div className="text-xs text-slate-400">Classification</div>
                    <div className="text-sm font-bold text-sky-300 mt-1">
                      {osteo.score >= 0.65 ? "High Severity" : osteo.score >= 0.38 ? "Moderate Risk" : "Low Risk / Normal"}
                    </div>
                  </div>
                </div>

                {/* Progress / Gauge Bar */}
                <div>
                  <div className="flex justify-between text-xs text-slate-400 mb-1 font-medium">
                    <span>Normal (0%)</span>
                    <span>Osteopenia (38%)</span>
                    <span>Osteoporosis (65%+)</span>
                  </div>
                  <div className="h-3 bg-slate-950 rounded-full p-0.5 border border-slate-800 relative overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        osteo.score >= 0.65 ? "bg-rose-500" : osteo.score >= 0.38 ? "bg-amber-500" : "bg-emerald-500"
                      }`}
                      style={{ width: `${Math.max(6, Math.min(100, (osteo.score ?? 0) * 100))}%` }}
                    />
                  </div>
                </div>

                {/* Clinical Notes / Findings Description */}
                <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800/80 text-slate-300 text-xs leading-relaxed space-y-1">
                  <div className="font-bold text-slate-200 uppercase tracking-wider text-[10px] text-sky-400">Clinical Explanation & Indicators</div>
                  <p>{osteo.clinical_notes || "Structural edge intensity and cortical attenuation measured."}</p>
                </div>
              </div>

              {/* Secondary Pathology Findings List */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 border-b border-slate-800 pb-2">
                  Multi-Pathology Secondary Findings
                </h3>
                <div className="space-y-3">
                  {Object.entries(inference.supporting_findings || {}).map(([finding, score]: [string, any]) => {
                    const scoreNum = typeof score === "number" ? score : 0;
                    const pct = Math.round(scoreNum * 100);
                    return (
                      <div key={finding} className="space-y-1">
                        <div className="flex justify-between text-xs font-medium">
                          <span className="text-slate-300">{finding}</span>
                          <span className="text-slate-400 font-mono">{pct}%</span>
                        </div>
                        <div className="h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800/80">
                          <div
                            className="h-full bg-sky-500/80 rounded-full"
                            style={{ width: `${Math.max(2, pct)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

            </div>

            {/* Right Column: Interactive XAI Explainable AI Visualizer (7 cols) */}
            <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl flex flex-col justify-between space-y-4">
              
              <div>
                <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-3 gap-2">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-emerald-400">Explainable AI Engine (XAI)</h3>
                    <h2 className="text-lg font-bold text-white">Grad-CAM Model Attention Saliency Map</h2>
                  </div>

                  {/* Mode Toggles */}
                  <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                    <button
                      onClick={() => setXaiMode("overlay")}
                      className={`px-3 py-1 text-xs font-bold rounded-md transition ${
                        xaiMode === "overlay" ? "bg-sky-600 text-white shadow" : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      XAI Overlay
                    </button>
                    <button
                      onClick={() => setXaiMode("heatmap")}
                      className={`px-3 py-1 text-xs font-bold rounded-md transition ${
                        xaiMode === "heatmap" ? "bg-sky-600 text-white shadow" : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Grad-CAM Heatmap
                    </button>
                    <button
                      onClick={() => setXaiMode("original")}
                      className={`px-3 py-1 text-xs font-bold rounded-md transition ${
                        xaiMode === "original" ? "bg-sky-600 text-white shadow" : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Raw X-Ray
                    </button>
                  </div>
                </div>

                {/* XAI Image Display Area */}
                <div className="mt-4 relative bg-black/60 rounded-xl border border-slate-800 p-3 min-h-[360px] flex items-center justify-center overflow-hidden">
                  {xaiMode === "overlay" && (
                    <img
                      src={
                        inference.overlay_path
                          ? `http://localhost:8000${inference.overlay_path}`
                          : inference.heatmap_path
                          ? `http://localhost:8000${inference.heatmap_path}`
                          : selectedImage
                          ? `http://localhost:8000${selectedImage}`
                          : ""
                      }
                      alt="XAI Grad-CAM Overlay"
                      className="max-h-[380px] w-auto object-contain rounded"
                    />
                  )}

                  {xaiMode === "heatmap" && (
                    <img
                      src={
                        inference.heatmap_path
                          ? `http://localhost:8000${inference.heatmap_path}`
                          : ""
                      }
                      alt="Grad-CAM Heatmap"
                      className="max-h-[380px] w-auto object-contain rounded"
                    />
                  )}

                  {xaiMode === "original" && selectedImage && (
                    <img
                      src={`http://localhost:8000${selectedImage}`}
                      alt="Raw Input X-Ray"
                      className="max-h-[380px] w-auto object-contain rounded"
                    />
                  )}
                </div>
              </div>

              {/* XAI Explanation Legend & Note */}
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-sky-400 uppercase tracking-wider">XAI Attention Heatmap Guide</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-slate-400">Low</span>
                    <div className="h-2.5 w-24 rounded-full bg-gradient-to-r from-blue-600 via-green-400 via-yellow-400 to-red-600 border border-slate-700"></div>
                    <span className="text-[10px] text-slate-400">High Attention</span>
                  </div>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  The Grad-CAM visual heatmap highlights anatomical regions driving the model's prediction. Warm regions (red/yellow) indicate key structural features (trabecular density variations, cortical boundaries, or fracture sites) influencing the Osteoporosis diagnostic risk output.
                </p>
              </div>

            </div>

          </div>
        ) : null}

    </div>
  );
}


