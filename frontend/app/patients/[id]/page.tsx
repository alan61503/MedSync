"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import UploadDropzone from "../../../components/UploadDropzone";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ViewMode = "overlay" | "side-by-side" | "heatmap" | "original";

export default function PatientPage() {
  const params = useParams();
  const id = Array.isArray(params?.id) ? params.id[0] : params?.id;

  const [patient, setPatient] = useState<any>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [inference, setInference] = useState<any>(null);
  const [loadingInference, setLoadingInference] = useState(false);

  // XAI Studio Radiologist Controls
  const [xaiMode, setXaiMode] = useState<ViewMode>("overlay");
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.65);
  const [brightness, setBrightness] = useState<number>(100);
  const [contrast, setContrast] = useState<number>(100);
  const [invertImage, setInvertImage] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  // Navigation & Clinical Notes
  const [activeTab, setActiveTab] = useState<"diagnostics" | "notes">("diagnostics");
  const [reportText, setReportText] = useState("");
  const [savingNote, setSavingNote] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const showStatus = (msg: string) => {
    setStatusMsg(msg);
    setTimeout(() => setStatusMsg(null), 4000);
  };

  const loadPatient = async () => {
    if (!id) return;
    try {
      const res = await fetch(`${API_BASE}/api/patients/${id}`);
      if (!res.ok) throw new Error("Patient not found");
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
      const fullUrl = imagePath.startsWith("http") ? imagePath : `${API_BASE}${imagePath}`;
      const res = await fetch(`${API_BASE}/api/run-inference`, {
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

  const deleteXrayImage = async (filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm(`Delete X-ray scan "${filename}"?`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/patients/${id}/xrays/${filename}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setSelectedImage(null);
        setInference(null);
        showStatus("X-ray image deleted.");
        await loadPatient();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const deleteReportFile = async (filename: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/patients/${id}/reports/${filename}`, {
        method: "DELETE",
      });
      if (res.ok) {
        showStatus("Note deleted.");
        await loadPatient();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleReportSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reportText.trim()) return;
    setSavingNote(true);
    try {
      const fd = new FormData();
      fd.append("text", reportText);
      const res = await fetch(`${API_BASE}/api/patients/${id}/upload-report`, {
        method: "POST",
        body: fd,
      });
      if (res.ok) {
        setReportText("");
        showStatus("Clinical note saved.");
        await loadPatient();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSavingNote(false);
    }
  };

  const generateAiObservation = () => {
    if (!inference || !inference.osteoporosis) return;
    const ost = inference.osteoporosis;
    const scorePct = ost.percentage || (ost.score * 100).toFixed(1);
    const text = `AUTOMATED AI DIAGNOSTIC REPORT:
Target Disease: Osteoporosis
Risk Level: ${ost.risk_level} (${scorePct}% probability)
Clinical Notes: ${ost.clinical_notes}

Key Findings:
- Cortical Bone Thinning: ${(inference.supporting_findings?.["Cortical Bone Thinning"] * 100 || 0).toFixed(1)}%
- Trabecular Degradation: ${(inference.supporting_findings?.["Trabecular Microarchitecture Degradation"] * 100 || 0).toFixed(1)}%
- BMD Attenuation: ${(inference.supporting_findings?.["Bone Mineral Density (BMD) Attenuation"] * 100 || 0).toFixed(1)}%
Recommendation: Physician review and follow-up DEXA scan scheduled.`;

    setReportText(text);
    showStatus("AI Diagnostic summary pre-filled!");
  };

  const resetViewControls = () => {
    setOverlayOpacity(0.65);
    setBrightness(100);
    setContrast(100);
    setInvertImage(false);
  };

  if (!patient) {
    return (
      <div className="min-h-[50vh] flex items-center justify-center">
        <div className="text-xs font-semibold text-indigo-600 flex items-center gap-2">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
          </svg>
          Loading patient profile...
        </div>
      </div>
    );
  }

  const osteo = inference?.osteoporosis || {};
  const isHighRisk = osteo.score >= 0.62;
  const isModerateRisk = osteo.score >= 0.35 && osteo.score < 0.62;

  const riskBadgeClass = isHighRisk
    ? "bg-rose-50 text-rose-700 border-rose-200"
    : isModerateRisk
    ? "bg-amber-50 text-amber-700 border-amber-200"
    : "bg-emerald-50 text-emerald-700 border-emerald-200";

  const dexaTScoreNum = osteo.score ? parseFloat((-1.0 - (osteo.score * 2.2)).toFixed(1)) : -1.2;
  const dexaTScoreStr = dexaTScoreNum.toFixed(1);

  // Position for T-Score Scale (-4.0 to +1.0)
  const tScoreGaugePos = Math.max(0, Math.min(100, ((dexaTScoreNum - (-4.0)) / 5.0) * 100));

  const baseImageFilterStyle = {
    filter: `${invertImage ? "invert(100%) " : ""}brightness(${brightness}%) contrast(${contrast}%)`,
  };

  const overlayImageFilterStyle = {
    filter: `brightness(${brightness}%) contrast(${contrast}%)`,
  };

  const activeHeatmap = inference?.heatmap_path ? `${API_BASE}${inference.heatmap_path}` : undefined;
  const activeOverlay = inference?.overlay_path ? `${API_BASE}${inference.overlay_path}` : undefined;

  const getGptExportText = () => {
    if (!patient || !inference) return "";
    const ost = inference.osteoporosis || {};
    const pct = ost.percentage || (ost.score * 100).toFixed(1);
    const scoreVal = ost.score || 0;
    const tScore = (-1.0 - (scoreVal * 2.2)).toFixed(1);
    const findings = inference.supporting_findings || {};
    const cThin = ((findings["Cortical Bone Thinning"] || 0) * 100).toFixed(1);
    const trabDeg = ((findings["Trabecular Microarchitecture Degradation"] || 0) * 100).toFixed(1);
    const bmdAtt = ((findings["Bone Mineral Density (BMD) Attenuation"] || 0) * 100).toFixed(1);
    const fracRisk = ((findings["Fragility Fracture Indicator"] || 0) * 100).toFixed(1);

    return `[MEDSYNC CLINICAL ATTRIBUTION EXPORT]
Patient Reference: ${patient.name} (${patient.id})
Demographics: Age ${patient.age} / Gender ${patient.gender}

PRIMARY AI FINDINGS:
- Target Condition: Osteoporosis
- AI Probability Score: ${pct}%
- Estimated DEXA T-Score: ${tScore} SD
- Calibrated Risk Level: ${ost.risk_level || "Pending"}

NEURAL NETWORK ATTRIBUTION METRICS:
- Cortical Bone Thinning: ${cThin}%
- Trabecular Microarchitecture Degradation: ${trabDeg}%
- Bone Mineral Density (BMD) Attenuation: ${bmdAtt}%
- Fragility Fracture Indicator: ${fracRisk}%

CLINICAL RECOMMENDATION SUMMARY:
"${ost.clinical_notes || "BMD attenuation analyzed."}"

---
VALIDATION PROMPT:
"Please review the clinical data above. Does the combination of cortical thinning (${cThin}%), trabecular degradation (${trabDeg}%), and BMD attenuation (${bmdAtt}%) align with an estimated DEXA T-Score of ${tScore} SD and a diagnosis of ${ost.risk_level}? Provide a validation checklist."`;
  };

  return (
    <div className="space-y-6 font-['Plus_Jakarta_Sans',sans-serif]">
      {/* Toast Notification */}
      {statusMsg && (
        <div className="fixed bottom-6 right-6 z-50 bg-slate-900 text-white px-4 py-2.5 rounded-xl shadow-xl text-xs font-semibold">
          {statusMsg}
        </div>
      )}

      {/* Patient Header Navigation Bar */}
      <div className="bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <Link
            href="/"
            className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-bold transition flex items-center gap-1"
          >
            ← Back
          </Link>

          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-xl font-extrabold text-slate-900 tracking-tight">{patient.name}</h1>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-slate-100 text-slate-600 border border-slate-200">
                {patient.id}
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Age: <strong className="text-slate-800">{patient.age}</strong> &bull; Gender: <strong className="text-slate-800">{patient.gender}</strong> &bull; Study: <strong className="text-indigo-600 font-semibold">Osteoporosis X-Ray & Grad-CAM XAI</strong>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => window.print()}
            className="no-print px-3.5 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-bold transition"
          >
            🖨️ Print Report
          </button>

          <button
            onClick={() => setActiveTab("diagnostics")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition ${
              activeTab === "diagnostics" ? "bg-indigo-600 text-white shadow-sm" : "bg-slate-100 text-slate-600 hover:text-slate-900"
            }`}
          >
            X-Ray Diagnostics
          </button>

          <button
            onClick={() => setActiveTab("notes")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition ${
              activeTab === "notes" ? "bg-indigo-600 text-white shadow-sm" : "bg-slate-100 text-slate-600 hover:text-slate-900"
            }`}
          >
            Clinical Notes ({patient.reports?.length || 0})
          </button>
        </div>
      </div>

      {activeTab === "diagnostics" ? (
        <>
          {/* Top Section: Upload Dropzone & Thumbnail Gallery */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            <div className="lg:col-span-1">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Upload Test Scan</h2>
              <UploadDropzone patientId={patient.id} onUploadComplete={loadPatient} />
            </div>

            <div className="lg:col-span-2 bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                    Uploaded X-Ray Scans ({patient.images?.length || 0})
                  </h2>
                  <span className="text-[11px] text-slate-400">Click scan to run AI prediction</span>
                </div>

                {!patient.images || patient.images.length === 0 ? (
                  <div className="p-8 text-center text-slate-400 border border-dashed border-slate-200 rounded-xl text-xs">
                    No X-ray pictures uploaded yet. Use the upload panel to add test scans.
                  </div>
                ) : (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 max-h-44 overflow-y-auto pr-1">
                    {patient.images.map((img: any) => {
                      const isSelected = selectedImage === img.file_path;
                      const imgUrl = img.file_path.startsWith("http")
                        ? img.file_path
                        : `${API_BASE}${img.file_path}`;
                      return (
                        <div
                          key={img.id}
                          onClick={() => {
                            setSelectedImage(img.file_path);
                            fetchInference(img.file_path);
                          }}
                          className={`relative group cursor-pointer rounded-xl p-2 border transition ${
                            isSelected
                              ? "bg-indigo-50/60 border-indigo-500 shadow-sm ring-1 ring-indigo-500"
                              : "bg-slate-50/60 border-slate-200 hover:border-slate-300"
                          }`}
                        >
                          <button
                            onClick={(e) => deleteXrayImage(img.filename, e)}
                            title="Delete Image"
                            className="absolute top-1 right-1 z-10 w-5 h-5 rounded-full bg-slate-800 text-white text-[10px] opacity-0 group-hover:opacity-100 hover:bg-red-600 transition flex items-center justify-center shadow"
                          >
                            ✕
                          </button>

                          <div className="h-16 bg-slate-100 rounded-lg flex items-center justify-center overflow-hidden">
                            <img src={imgUrl} alt={img.filename} className="object-contain h-full w-full" />
                          </div>
                          <div className="mt-1 text-[11px] font-medium text-slate-700 truncate">{img.filename}</div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {selectedImage && (
                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between">
                  <span className="text-xs text-slate-500 truncate">Selected: <strong className="text-slate-800">{selectedImage.split("/").pop()}</strong></span>
                  <button
                    onClick={() => fetchInference(selectedImage)}
                    disabled={loadingInference}
                    className="px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg transition shadow-sm flex items-center gap-1.5"
                  >
                    {loadingInference ? "Analyzing..." : "Re-Run AI Inference & XAI"}
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Main AI Results & Visualizer */}
          {loadingInference ? (
            <div className="bg-white border border-slate-200/80 rounded-2xl p-12 text-center shadow-sm">
              <div className="inline-flex items-center gap-2.5 px-4 py-2 rounded-full bg-indigo-50 text-indigo-700 font-semibold text-xs border border-indigo-200">
                <svg className="animate-spin h-4 w-4 text-indigo-600" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                </svg>
                Computing Osteoporosis AI Inference & Grad-CAM Heatmaps...
              </div>
            </div>
          ) : inference ? (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              
              {/* Left Column: Osteoporosis Primary Prediction Card (5 cols) */}
              <div className="lg:col-span-5 space-y-5">
                
                <div className="bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                    <div>
                      <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-600">Target Disease</span>
                      <h2 className="text-lg font-extrabold text-slate-900">Osteoporosis Diagnosis</h2>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold border ${riskBadgeClass}`}>
                      {osteo.risk_level || "Analyzed"}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200/60">
                      <div className="text-[11px] font-medium text-slate-500">Osteoporosis Risk</div>
                      <div className="text-2xl font-black text-slate-900 mt-0.5">
                        {osteo.percentage ?? ((osteo.score ?? 0) * 100).toFixed(1)}%
                      </div>
                    </div>

                    <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200/60">
                      <div className="text-[11px] font-medium text-slate-500">Estimated DEXA T-Score</div>
                      <div className={`text-2xl font-black mt-0.5 font-mono ${dexaTScoreNum <= -2.5 ? "text-rose-600" : dexaTScoreNum <= -1.0 ? "text-amber-600" : "text-emerald-600"}`}>
                        {dexaTScoreStr} SD
                      </div>
                    </div>
                  </div>

                  {/* DEXA T-Score Reference Visual Range Bar */}
                  <div className="space-y-1.5 bg-slate-50 p-3.5 rounded-xl border border-slate-200/60">
                    <div className="flex justify-between text-[11px] font-semibold text-slate-600">
                      <span>DEXA T-Score Scale</span>
                      <span className="font-mono font-bold text-indigo-600">{dexaTScoreStr} SD</span>
                    </div>

                    <div className="relative h-3 rounded-full bg-gradient-to-r from-rose-500 via-amber-400 to-emerald-500 overflow-hidden">
                      <div
                        className="absolute top-0 bottom-0 w-1 bg-slate-900 shadow rounded-full transition-all duration-300 ring-2 ring-white"
                        style={{ left: `${tScoreGaugePos}%` }}
                      ></div>
                    </div>

                    <div className="flex justify-between text-[9px] font-semibold text-slate-400 pt-0.5">
                      <span>Osteoporosis (≤ -2.5)</span>
                      <span>Osteopenia (-1.0)</span>
                      <span>Normal</span>
                    </div>
                  </div>

                  <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200/60 text-xs text-slate-600 leading-relaxed">
                    <div className="font-bold text-slate-800 text-[11px] mb-0.5">Diagnostic Explanation:</div>
                    <p>{osteo.clinical_notes || "Bone Mineral Density (BMD) attenuation analyzed."}</p>
                  </div>
                </div>

                <div className="bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm space-y-3">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 border-b border-slate-100 pb-2">
                    Secondary Findings
                  </h3>
                  <div className="space-y-2.5">
                    {Object.entries(inference.supporting_findings || {}).map(([finding, score]: [string, any]) => {
                      const scoreNum = typeof score === "number" ? score : 0;
                      const pct = Math.round(scoreNum * 100);
                      return (
                        <div key={finding} className="space-y-1">
                          <div className="flex justify-between text-xs font-medium">
                            <span className="text-slate-700">{finding}</span>
                            <span className="text-slate-500 font-mono font-bold">{pct}%</span>
                          </div>
                          <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-indigo-600 rounded-full transition-all duration-300"
                              style={{ width: `${Math.max(2, pct)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* GPT Validation Export (2-Factor Check) */}
                <div className="bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      GPT Validation Export (2-Factor Check)
                    </h3>
                    <button
                      type="button"
                      onClick={() => {
                        const text = getGptExportText();
                        navigator.clipboard.writeText(text);
                        showStatus("Clinical export copied to clipboard!");
                      }}
                      className="px-2.5 py-1 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-[10px] font-bold rounded-lg transition animate-pulse"
                    >
                      📋 Copy Text
                    </button>
                  </div>
                  <p className="text-[11px] text-slate-500 leading-normal">
                    Copy this structured summary to paste into ChatGPT/Claude to run a secondary clinical verification.
                  </p>
                  <textarea
                    readOnly
                    rows={6}
                    value={getGptExportText()}
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-[10px] font-mono text-slate-700 focus:outline-none resize-none"
                  />
                </div>

              </div>

              {/* Right Column: Clean XAI Studio (7 cols) */}
              <div className="lg:col-span-7 bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm flex flex-col justify-between space-y-4">
                
                <div className="space-y-3">
                  {/* Viewer Modes Header */}
                  <div className="flex flex-wrap items-center justify-between border-b border-slate-100 pb-3 gap-2">
                    <div>
                      <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-600">Explainable AI (XAI)</span>
                      <h2 className="text-base font-bold text-slate-900">Grad-CAM Feature Heatmap</h2>
                    </div>

                    <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg">
                      <button
                        onClick={() => setXaiMode("overlay")}
                        className={`px-3 py-1 text-xs font-bold rounded-md transition ${
                          xaiMode === "overlay" ? "bg-white text-indigo-600 shadow-sm" : "text-slate-600 hover:text-slate-900"
                        }`}
                      >
                        Overlay
                      </button>
                      <button
                        onClick={() => setXaiMode("side-by-side")}
                        className={`px-3 py-1 text-xs font-bold rounded-md transition ${
                          xaiMode === "side-by-side" ? "bg-white text-indigo-600 shadow-sm" : "text-slate-600 hover:text-slate-900"
                        }`}
                      >
                        Dual View
                      </button>
                      <button
                        onClick={() => setXaiMode("heatmap")}
                        className={`px-3 py-1 text-xs font-bold rounded-md transition ${
                          xaiMode === "heatmap" ? "bg-white text-indigo-600 shadow-sm" : "text-slate-600 hover:text-slate-900"
                        }`}
                      >
                        Heatmap
                      </button>
                      <button
                        onClick={() => setXaiMode("original")}
                        className={`px-3 py-1 text-xs font-bold rounded-md transition ${
                          xaiMode === "original" ? "bg-white text-indigo-600 shadow-sm" : "text-slate-600 hover:text-slate-900"
                        }`}
                      >
                        Original
                      </button>
                    </div>
                  </div>

                  {/* Radiologist Adjustments Toolbar */}
                  <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60 space-y-2.5">
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-slate-600">
                      {xaiMode === "overlay" && (
                        <div className="space-y-1">
                          <div className="flex justify-between font-semibold">
                            <span>Opacity:</span>
                            <span className="font-mono text-indigo-600 font-bold">{Math.round(overlayOpacity * 100)}%</span>
                          </div>
                          <input
                            type="range"
                            min="0.1"
                            max="1.0"
                            step="0.05"
                            value={overlayOpacity}
                            onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                            className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                          />
                        </div>
                      )}

                      <div className="space-y-1">
                        <div className="flex justify-between font-semibold">
                          <span>Brightness:</span>
                          <span className="font-mono text-indigo-600 font-bold">{brightness}%</span>
                        </div>
                        <input
                          type="range"
                          min="50"
                          max="150"
                          step="5"
                          value={brightness}
                          onChange={(e) => setBrightness(parseInt(e.target.value))}
                          className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                        />
                      </div>

                      <div className="space-y-1">
                        <div className="flex justify-between font-semibold">
                          <span>Contrast:</span>
                          <span className="font-mono text-indigo-600 font-bold">{contrast}%</span>
                        </div>
                        <input
                          type="range"
                          min="50"
                          max="150"
                          step="5"
                          value={contrast}
                          onChange={(e) => setContrast(parseInt(e.target.value))}
                          className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                        />
                      </div>
                    </div>

                    <div className="pt-2 border-t border-slate-200/60 flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setInvertImage(!invertImage)}
                          className={`px-2.5 py-1 rounded-md font-bold transition ${
                            invertImage
                              ? "bg-indigo-600 text-white"
                              : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-100"
                          }`}
                        >
                          🌓 Invert X-Ray
                        </button>
                        <button
                          onClick={resetViewControls}
                          className="px-2.5 py-1 bg-white hover:bg-slate-100 text-slate-600 rounded-md font-bold border border-slate-200 transition"
                        >
                          🔄 Reset
                        </button>
                      </div>

                      <button
                        onClick={() => setIsFullscreen(true)}
                        className="px-2.5 py-1 bg-white hover:bg-slate-100 text-indigo-600 font-bold rounded-md border border-slate-200 transition"
                      >
                        🔍 Fullscreen Inspection
                      </button>
                    </div>
                  </div>

                  {/* Main Viewer Display Box */}
                  <div className="mt-3 relative bg-slate-900 rounded-xl border border-slate-200 p-2 min-h-[360px] flex items-center justify-center overflow-hidden">
                    
                    {/* Overlay Mode */}
                    {xaiMode === "overlay" && selectedImage && (
                      <div className="relative max-h-[360px] inline-flex items-center justify-center">
                        <img
                          src={`${API_BASE}${selectedImage}`}
                          alt="Base Scan"
                          style={baseImageFilterStyle}
                          className="max-h-[360px] w-auto object-contain rounded"
                        />
                        <img
                          src={activeOverlay || activeHeatmap}
                          alt="XAI Overlay"
                          style={{
                            ...overlayImageFilterStyle,
                            opacity: overlayOpacity,
                          }}
                          className="absolute inset-0 w-full h-full object-contain rounded transition-opacity duration-150 pointer-events-none"
                        />
                      </div>
                    )}

                    {/* Dual Mode */}
                    {xaiMode === "side-by-side" && (
                      <div className="grid grid-cols-2 gap-2 w-full max-h-[360px]">
                        <div className="flex flex-col items-center">
                          <span className="text-[10px] font-bold text-slate-300 mb-1">Original Scan</span>
                          {selectedImage && (
                            <img
                              src={`${API_BASE}${selectedImage}`}
                              alt="Original"
                              style={baseImageFilterStyle}
                              className="max-h-[320px] object-contain rounded border border-slate-700"
                            />
                          )}
                        </div>
                        <div className="flex flex-col items-center">
                          <span className="text-[10px] font-bold text-indigo-400 mb-1">XAI Heatmap</span>
                          <img
                            src={activeHeatmap}
                            alt="Heatmap"
                            style={overlayImageFilterStyle}
                            className="max-h-[320px] object-contain rounded border border-slate-700"
                          />
                        </div>
                      </div>
                    )}

                    {/* Heatmap Mode */}
                    {xaiMode === "heatmap" && (
                      <img
                        src={activeHeatmap}
                        alt="Heatmap Only"
                        style={overlayImageFilterStyle}
                        className="max-h-[360px] w-auto object-contain rounded"
                      />
                    )}

                    {/* Original Mode */}
                    {xaiMode === "original" && selectedImage && (
                      <img
                        src={`${API_BASE}${selectedImage}`}
                        alt="Original Scan"
                        style={baseImageFilterStyle}
                        className="max-h-[360px] w-auto object-contain rounded"
                      />
                    )}
                  </div>
                </div>

                <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200/60 text-xs text-slate-500 leading-relaxed">
                  <span className="font-bold text-slate-800">XAI Visual Attention:</span> Warm regions (red/yellow) indicate key anatomical features driving the Osteoporosis AI prediction.
                </div>

              </div>

            </div>
          ) : null}
        </>
      ) : (
        /* Clinical Notes Tab */
        <div className="bg-white border border-slate-200/80 rounded-2xl p-6 shadow-sm space-y-5">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h2 className="text-base font-bold text-slate-900">Clinical Notes</h2>

            {inference && (
              <button
                onClick={generateAiObservation}
                className="px-3.5 py-1.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 text-xs font-bold rounded-lg transition"
              >
                ⚡ Auto-Generate AI Diagnostic Report
              </button>
            )}
          </div>
          
          <form onSubmit={handleReportSubmit} className="space-y-3">
            <textarea
              rows={4}
              value={reportText}
              onChange={(e) => setReportText(e.target.value)}
              placeholder="Enter DEXA scan notes or physician observations..."
              className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3.5 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-indigo-500 transition"
            />
            <button
              type="submit"
              disabled={savingNote}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg transition shadow-sm"
            >
              {savingNote ? "Saving..." : "Save Note"}
            </button>
          </form>

          <div className="space-y-2 pt-3 border-t border-slate-100">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">Saved Notes ({patient.reports?.length || 0})</h3>
            {patient.reports && patient.reports.length > 0 ? (
              <div className="space-y-2">
                {patient.reports.map((r: any, idx: number) => (
                  <div key={idx} className="bg-slate-50 p-3 rounded-xl border border-slate-200/60 text-xs text-slate-700 flex items-center justify-between">
                    <span className="font-mono">{r.filename}</span>
                    <div className="flex items-center gap-3">
                      <a href={`${API_BASE}${r.path}`} target="_blank" rel="noreferrer" className="text-indigo-600 font-bold hover:underline">
                        View Note →
                      </a>
                      <button
                        onClick={() => deleteReportFile(r.filename)}
                        title="Delete Note"
                        className="text-slate-400 hover:text-red-600 transition"
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-400">No notes attached yet.</p>
            )}
          </div>
        </div>
      )}

      {/* Fullscreen Lightbox Inspection Modal */}
      {isFullscreen && (
        <div className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md flex flex-col items-center justify-between p-6">
          <div className="w-full flex items-center justify-between text-xs text-slate-300 border-b border-slate-800 pb-3">
            <span className="font-bold text-white">Full-Screen Radiologist XAI Inspection</span>
            <button
              onClick={() => setIsFullscreen(false)}
              className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-white font-bold rounded-xl transition"
            >
              Close (Esc) ✕
            </button>
          </div>

          <div className="relative max-h-[85vh] flex items-center justify-center my-auto">
            {selectedImage && (
              <div className="relative max-h-[80vh] inline-flex items-center justify-center">
                <img
                  src={`${API_BASE}${selectedImage}`}
                  alt="Fullscreen Scan"
                  style={baseImageFilterStyle}
                  className="max-h-[80vh] w-auto object-contain rounded border border-slate-800"
                />
                {xaiMode === "overlay" && (
                  <img
                    src={activeOverlay || activeHeatmap}
                    alt="Fullscreen XAI Overlay"
                    style={{
                      ...overlayImageFilterStyle,
                      opacity: overlayOpacity,
                    }}
                    className="absolute inset-0 w-full h-full object-contain rounded transition-opacity duration-150 pointer-events-none"
                  />
                )}
              </div>
            )}
          </div>

          <div className="text-xs text-slate-400 font-medium">
            Use Brightness ({brightness}%) & Contrast ({contrast}%) controls to adjust scan clarity.
          </div>
        </div>
      )}

    </div>
  );
}
