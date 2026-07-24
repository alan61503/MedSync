"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";

type Summary = {
  id: string;
  name: string;
  age?: number;
  gender?: string;
  xrays: number;
  reports: number;
  completion: number;
  risk_level?: string;
  risk_score?: number | null;
  t_score?: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Page() {
  const [patients, setPatients] = useState<Summary[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [riskFilter, setRiskFilter] = useState<"ALL" | "High Risk" | "Moderate Risk" | "Low Risk">("ALL");
  const [sortBy, setSortBy] = useState<"name" | "risk" | "age" | "scans">("name");
  
  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [form, setForm] = useState({ name: "", age: "", gender: "Female" });
  const [statusMsg, setStatusMsg] = useState<{ message: string; type: "success" | "error" } | null>(null);

  const showStatus = (message: string, type: "success" | "error" = "success") => {
    setStatusMsg({ message, type });
    setTimeout(() => setStatusMsg(null), 4000);
  };

  const loadPatients = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/api/patients/summary`);
      const data = await response.json();
      setPatients(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error("Failed to load patients", err);
      showStatus("Could not connect to backend server.", "error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPatients();
  }, []);

  const createPatient = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name.trim()) return;
    setCreating(true);
    try {
      const payload = {
        name: form.name.trim(),
        age: form.age ? parseInt(form.age) : 48,
        gender: form.gender,
      };
      const res = await fetch(`${API_BASE}/api/patients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const newPatient = await res.json();
        const summaryItem: Summary = {
          id: newPatient.id,
          name: newPatient.name,
          age: newPatient.age,
          gender: newPatient.gender,
          xrays: 0,
          reports: 0,
          completion: 25,
          risk_level: "Pending Scan",
          t_score: -1.2,
        };
        setPatients((prev) => [summaryItem, ...prev]);
        setForm({ name: "", age: "", gender: "Female" });
        showStatus(`Registered patient "${newPatient.name}"`);
      } else {
        showStatus("Failed to create patient record.", "error");
      }
    } catch (err) {
      console.error(err);
      showStatus("Error connecting to backend.", "error");
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (patientId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();

    setDeletingId(patientId);
    setPatients((prev) => prev.filter((p) => p.id !== patientId));

    try {
      await fetch(`${API_BASE}/api/patients/${patientId}`, {
        method: "DELETE",
      });
      showStatus("Patient record deleted.");
    } catch (err) {
      console.error("Delete error:", err);
      loadPatients();
    } finally {
      setDeletingId(null);
    }
  };

  // Metrics
  const totalPatients = patients.length;
  const highRiskCount = patients.filter((p) => p.risk_level === "High Risk").length;
  const totalScans = patients.reduce((acc, p) => acc + (p.xrays || 0), 0);

  // Filter & Sort
  const filteredPatients = patients
    .filter((p) => {
      const matchesSearch =
        p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.id.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesRisk = riskFilter === "ALL" || p.risk_level === riskFilter;
      return matchesSearch && matchesRisk;
    })
    .sort((a, b) => {
      if (sortBy === "name") return a.name.localeCompare(b.name);
      if (sortBy === "age") return (b.age || 0) - (a.age || 0);
      if (sortBy === "scans") return (b.xrays || 0) - (a.xrays || 0);
      if (sortBy === "risk") return (b.risk_score || 0) - (a.risk_score || 0);
      return 0;
    });

  return (
    <div className="space-y-6 font-['Plus_Jakarta_Sans',sans-serif]">
      
      {/* Top Header & Minimalist Add Patient Form */}
      <div className="bg-white border border-slate-200/80 rounded-2xl p-6 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-extrabold text-slate-900 tracking-tight">Patient Directory</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Osteoporosis diagnostic research cohort & Grad-CAM XAI saliency engine
          </p>
        </div>

        {/* Inline Add Patient Form */}
        <form onSubmit={createPatient} className="flex flex-wrap items-center gap-2">
          <input
            required
            placeholder="Patient Name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-indigo-500 transition"
          />
          <input
            placeholder="Age"
            type="number"
            value={form.age}
            onChange={(e) => setForm({ ...form, age: e.target.value })}
            className="bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-xs text-slate-800 placeholder-slate-400 w-16 focus:outline-none focus:border-indigo-500 transition"
          />
          <select
            value={form.gender}
            onChange={(e) => setForm({ ...form, gender: e.target.value })}
            className="bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-xs text-slate-700 focus:outline-none focus:border-indigo-500 transition cursor-pointer"
          >
            <option>Female</option>
            <option>Male</option>
          </select>
          <button
            type="submit"
            disabled={creating}
            className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg transition shadow-sm"
          >
            {creating ? "Adding..." : "+ Add Patient"}
          </button>
        </form>
      </div>

      {statusMsg && (
        <div
          className={`px-4 py-2.5 rounded-xl text-xs font-medium border ${
            statusMsg.type === "error"
              ? "bg-red-50 text-red-700 border-red-200"
              : "bg-emerald-50 text-emerald-700 border-emerald-200"
          }`}
        >
          {statusMsg.message}
        </div>
      )}

      {/* Cohort Stats Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200/80 rounded-2xl p-4 shadow-sm flex items-center justify-between">
          <div>
            <div className="text-[11px] font-semibold text-slate-500">Total Cohort</div>
            <div className="text-xl font-extrabold text-slate-900 mt-0.5">{totalPatients} Patients</div>
          </div>
          <span className="text-xl">👥</span>
        </div>

        <div className="bg-white border border-slate-200/80 rounded-2xl p-4 shadow-sm flex items-center justify-between">
          <div>
            <div className="text-[11px] font-semibold text-slate-500">High Risk Cases</div>
            <div className="text-xl font-extrabold text-rose-600 mt-0.5">{highRiskCount}</div>
          </div>
          <span className="text-xl">⚠️</span>
        </div>

        <div className="bg-white border border-slate-200/80 rounded-2xl p-4 shadow-sm flex items-center justify-between">
          <div>
            <div className="text-[11px] font-semibold text-slate-500">X-Ray Scans</div>
            <div className="text-xl font-extrabold text-indigo-600 mt-0.5">{totalScans}</div>
          </div>
          <span className="text-xl">📷</span>
        </div>

        <div className="bg-white border border-slate-200/80 rounded-2xl p-4 shadow-sm flex items-center justify-between">
          <div>
            <div className="text-[11px] font-semibold text-slate-500">Diagnostic Status</div>
            <div className="text-xs font-bold text-emerald-600 mt-1 flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Active Engine
            </div>
          </div>
          <span className="text-xl">⚡</span>
        </div>
      </div>

      {/* Search Bar & Filter Controls */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 bg-white border border-slate-200/80 p-3.5 rounded-2xl shadow-sm">
        
        {/* Risk Filter Buttons */}
        <div className="flex items-center gap-1.5 overflow-x-auto w-full sm:w-auto">
          <button
            onClick={() => setRiskFilter("ALL")}
            className={`px-3 py-1 rounded-lg text-xs font-bold transition ${
              riskFilter === "ALL"
                ? "bg-indigo-600 text-white shadow-sm"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            All ({patients.length})
          </button>
          <button
            onClick={() => setRiskFilter("High Risk")}
            className={`px-3 py-1 rounded-lg text-xs font-bold transition ${
              riskFilter === "High Risk"
                ? "bg-rose-600 text-white shadow-sm"
                : "bg-rose-50 text-rose-700 hover:bg-rose-100"
            }`}
          >
            High Risk
          </button>
          <button
            onClick={() => setRiskFilter("Moderate Risk")}
            className={`px-3 py-1 rounded-lg text-xs font-bold transition ${
              riskFilter === "Moderate Risk"
                ? "bg-amber-600 text-white shadow-sm"
                : "bg-amber-50 text-amber-700 hover:bg-amber-100"
            }`}
          >
            Moderate
          </button>
          <button
            onClick={() => setRiskFilter("Low Risk")}
            className={`px-3 py-1 rounded-lg text-xs font-bold transition ${
              riskFilter === "Low Risk"
                ? "bg-emerald-600 text-white shadow-sm"
                : "bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
            }`}
          >
            Low Risk
          </button>
        </div>

        {/* Search & Sort */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <input
            type="text"
            placeholder="Search name or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full sm:w-48 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-indigo-500 transition"
          />

          <select
            value={sortBy}
            onChange={(e: any) => setSortBy(e.target.value)}
            className="bg-slate-50 border border-slate-200 px-2.5 py-1.5 rounded-lg text-xs text-slate-700 focus:outline-none focus:border-indigo-500 transition cursor-pointer"
          >
            <option value="name">Sort: Name</option>
            <option value="risk">Sort: Risk</option>
            <option value="scans">Sort: Scans</option>
            <option value="age">Sort: Age</option>
          </select>
        </div>

      </div>

      {/* Patients Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm animate-pulse h-40"></div>
          ))}
        </div>
      ) : filteredPatients.length === 0 ? (
        <div className="bg-white border border-slate-200/80 rounded-2xl p-12 text-center text-slate-400">
          <p className="text-xs font-medium">No patient records found.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filteredPatients.map((p) => {
            const isHigh = p.risk_level === "High Risk";
            const isMod = p.risk_level === "Moderate Risk";

            const badgeClass = isHigh
              ? "bg-rose-50 text-rose-700 border-rose-200"
              : isMod
              ? "bg-amber-50 text-amber-700 border-amber-200"
              : p.risk_level === "Low Risk"
              ? "bg-emerald-50 text-emerald-700 border-emerald-200"
              : "bg-slate-100 text-slate-600 border-slate-200";

            return (
              <div
                key={p.id}
                className="bg-white border border-slate-200/80 hover:border-indigo-300 rounded-2xl p-5 shadow-sm hover:shadow-md transition flex flex-col justify-between space-y-4 group"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h3 className="font-bold text-slate-900 text-base group-hover:text-indigo-600 transition">
                      {p.name}
                    </h3>
                    <div className="text-xs text-slate-500 mt-0.5">
                      Age: {p.age || 50} &bull; Gender: {p.gender || "Female"} &bull; ID: <span className="font-mono text-slate-400">{p.id}</span>
                    </div>
                  </div>

                  <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${badgeClass}`}>
                    {p.risk_level || "Pending"}
                  </span>
                </div>

                <div className="pt-3 border-t border-slate-100 flex items-center justify-between">
                  <div className="text-xs text-slate-500 font-medium flex items-center gap-3">
                    <span>📷 {p.xrays} Scan{p.xrays !== 1 ? "s" : ""}</span>
                    {p.t_score !== undefined && (
                      <span className="font-mono text-slate-600">T: {p.t_score} SD</span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={(e) => handleDelete(p.id, e)}
                      disabled={deletingId === p.id}
                      title="Delete Patient"
                      className="px-2 py-1 rounded-lg text-xs font-bold text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition"
                    >
                      {deletingId === p.id ? "..." : "Delete"}
                    </button>

                    <Link
                      href={`/patients/${p.id}`}
                      className="px-3 py-1.5 bg-slate-100 hover:bg-indigo-600 text-slate-700 hover:text-white text-xs font-semibold rounded-lg transition"
                    >
                      View Diagnosis →
                    </Link>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

    </div>
  );
}
