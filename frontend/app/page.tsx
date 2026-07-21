"use client";
import React, { useEffect, useState } from "react";

type Summary = {
  id: string;
  name: string;
  xrays: number;
  ct: number;
  mri: number;
  reports: number;
  completion: number;
};

export default function Page() {
  const [patients, setPatients] = useState<Summary[]>([]);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ name: "", age: "", gender: "", medical_history: "", previous_diseases: "", symptoms: "", notes: "" });

  const loadPatients = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/patients/summary");
      const data = await response.json();
      setPatients(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error("Failed to load patients", err);
    }
  };

  useEffect(() => {
    loadPatients().catch(console.error);
  }, []);

  const createPatient = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setError(null);
    try {
      const payload = { ...form, age: form.age ? parseInt(form.age) : undefined };
      const res = await fetch("http://localhost:8000/api/patients", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        throw new Error("Failed to create patient");
      }

      setForm({ name: "", age: "", gender: "", medical_history: "", previous_diseases: "", symptoms: "", notes: "" });
      await loadPatients();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to reach the backend service");
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="space-y-6 font-sans">
      {/* Header Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-extrabold text-white tracking-tight">Patient Directory</h1>
            <span className="px-3 py-1 bg-sky-500/20 text-sky-400 border border-sky-500/30 rounded-full text-xs font-bold uppercase tracking-wider">
              Research Cohort
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Select a patient to upload X-rays, run Osteoporosis risk model inference, and inspect Grad-CAM XAI heatmaps.
          </p>
        </div>

        <form onSubmit={createPatient} className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-wrap items-center gap-2">
          <input required placeholder="Patient Name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} className="bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-sky-500" />
          <input placeholder="Age" value={form.age} onChange={(e) => setForm({ ...form, age: e.target.value })} className="bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg text-xs text-white placeholder-slate-500 w-16 focus:outline-none focus:border-sky-500" />
          <select value={form.gender} onChange={(e) => setForm({ ...form, gender: e.target.value })} className="bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg text-xs text-white focus:outline-none focus:border-sky-500">
            <option value="">Gender</option>
            <option>Male</option>
            <option>Female</option>
            <option>Other</option>
          </select>
          <button type="submit" disabled={creating} className="px-4 py-1.5 bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold rounded-lg transition shadow">
            {creating ? "Creating..." : "+ New Patient"}
          </button>
        </form>
      </div>

      {error ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-xs font-medium text-red-400">
          {error}
        </div>
      ) : null}

      {/* Patients Grid */}
      <div>
        <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Clinical Cohort ({patients.length})</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {patients.map((p) => (
            <div key={p.id} className="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-2xl p-6 shadow-xl transition flex flex-col justify-between space-y-4">
              <div>
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-white">{p.name}</h3>
                  <span className="text-xs font-mono bg-slate-950 px-2.5 py-1 rounded border border-slate-800 text-slate-400">ID: {p.id}</span>
                </div>
                
                <div className="mt-4 grid grid-cols-2 gap-2 text-xs text-slate-400 bg-slate-950 p-3 rounded-xl border border-slate-800/60">
                  <div>X-Rays: <span className="font-semibold text-slate-200">{p.xrays}</span></div>
                  <div>XAI Status: <span className="font-semibold text-emerald-400">Ready</span></div>
                  <div>Focus: <span className="font-semibold text-sky-400">Osteoporosis</span></div>
                  <div>Reports: <span className="font-semibold text-slate-200">{p.reports}</span></div>
                </div>
              </div>

              <div>
                <div className="space-y-1 mb-3">
                  <div className="flex justify-between text-[11px] text-slate-400 font-medium">
                    <span>Study Progress</span>
                    <span>{p.completion}%</span>
                  </div>
                  <div className="h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                    <div className="h-full bg-sky-500 rounded-full" style={{ width: `${Math.max(8, p.completion)}%` }} />
                  </div>
                </div>

                <a
                  href={`/patients/${p.id}`}
                  className="w-full py-2.5 bg-slate-800 hover:bg-sky-600 text-slate-200 hover:text-white text-xs font-semibold rounded-lg transition shadow flex items-center justify-center gap-1.5 border border-slate-700 hover:border-sky-500"
                >
                  Open Patient X-Ray & XAI Visualizer →
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


