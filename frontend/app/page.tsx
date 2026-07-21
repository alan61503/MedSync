"use client";
import React, { useEffect, useState } from "react";

type Summary = {
  id: string;
  name: string;
  age?: number;
  gender?: string;
  xrays: number;
  reports: number;
  completion: number;
};

export default function Page() {
  const [patients, setPatients] = useState<Summary[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [form, setForm] = useState({ name: "", age: "", gender: "Female" });

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
    loadPatients();
  }, []);

  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const createPatient = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name.trim()) return;
    setCreating(true);
    setStatusMsg(null);
    try {
      const payload = { name: form.name.trim(), age: form.age ? parseInt(form.age) : 48, gender: form.gender };
      const res = await fetch("http://localhost:8000/api/patients", {
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
        };
        // Optimistically add to state
        setPatients((prev) => [summaryItem, ...prev]);
        setForm({ name: "", age: "", gender: "Female" });
        setStatusMsg(`Successfully registered patient "${newPatient.name}"`);
        setTimeout(() => setStatusMsg(null), 4000);
      } else {
        setStatusMsg("Failed to create patient on backend server.");
      }
    } catch (err) {
      console.error(err);
      setStatusMsg("Error connecting to Flask backend server.");
    } finally {
      setCreating(false);
    }
  };


  const handleDelete = async (patientId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();

    // Optimistically update UI
    setDeletingId(patientId);
    setPatients((prev) => prev.filter((p) => p.id !== patientId));

    try {
      await fetch(`http://localhost:8000/api/patients/${patientId}`, {
        method: "DELETE",
      });
    } catch (err) {
      console.error("Delete error:", err);
      // reload if error
      loadPatients();
    } finally {
      setDeletingId(null);
    }
  };

  const filteredPatients = patients.filter(
    (p) =>
      p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6 font-['Plus_Jakarta_Sans',sans-serif]">
      
      {/* Top Header & Registration Form */}
      <div className="bg-white border border-slate-200/80 rounded-2xl p-6 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-extrabold text-slate-900 tracking-tight">Patient Directory</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Osteoporosis diagnostic research cohort and XAI saliency analysis
          </p>
        </div>

        {/* Simple Add Patient Form */}
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
            className="bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-xs text-slate-700 focus:outline-none focus:border-indigo-500 transition"
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
        <div className={`px-4 py-2.5 rounded-xl text-xs font-medium border ${statusMsg.includes("Failed") || statusMsg.includes("Error") ? "bg-red-50 text-red-700 border-red-200" : "bg-emerald-50 text-emerald-700 border-emerald-200"}`}>
          {statusMsg}
        </div>
      )}


      {/* Search Bar & List Subheader */}
      <div className="flex items-center justify-between gap-4">
        <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Cohort Patients ({filteredPatients.length})
        </h2>

        <div className="w-full sm:w-64">
          <input
            type="text"
            placeholder="Search patient name or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-white border border-slate-200 px-3.5 py-1.5 rounded-xl text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-indigo-500 transition shadow-sm"
          />
        </div>
      </div>

      {/* Minimalist Patients Grid */}
      {filteredPatients.length === 0 ? (
        <div className="bg-white border border-slate-200/80 rounded-2xl p-12 text-center text-slate-400">
          <p className="text-xs font-medium">No patient records found.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filteredPatients.map((p) => (
            <div
              key={p.id}
              className="bg-white border border-slate-200/80 hover:border-indigo-300 rounded-2xl p-5 shadow-sm hover:shadow-md transition flex flex-col justify-between space-y-4 group"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h3 className="font-bold text-slate-900 text-base group-hover:text-indigo-600 transition">{p.name}</h3>
                  <div className="text-xs text-slate-500 mt-0.5">
                    Age: {p.age || 50} &bull; Gender: {p.gender || "Female"} &bull; ID: <span className="font-mono text-slate-400">{p.id}</span>
                  </div>
                </div>

                {/* Instant Delete Button */}
                <button
                  onClick={(e) => handleDelete(p.id, e)}
                  disabled={deletingId === p.id}
                  title="Delete Patient"
                  className="px-2 py-1 rounded-lg text-xs font-bold text-slate-400 hover:text-red-600 hover:bg-red-50 transition border border-transparent hover:border-red-100"
                >
                  {deletingId === p.id ? "Deleting..." : "Delete"}
                </button>
              </div>

              <div className="pt-3 border-t border-slate-100 flex items-center justify-between">
                <span className="text-xs text-slate-500 font-medium">
                  📷 {p.xrays} Scan{p.xrays !== 1 ? "s" : ""}
                </span>
                <a
                  href={`/patients/${p.id}`}
                  className="px-3 py-1.5 bg-slate-100 hover:bg-indigo-600 text-slate-700 hover:text-white text-xs font-semibold rounded-lg transition"
                >
                  View Diagnosis →
                </a>
              </div>
            </div>
          ))}
        </div>
      )}

    </div>
  );
}




