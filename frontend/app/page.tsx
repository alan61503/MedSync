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
    const response = await fetch("http://localhost:8000/api/patients/summary");
    const data = await response.json();
    setPatients(Array.isArray(data) ? data : []);
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
    <div className="container mx-auto p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">MedSync Dashboard</h1>
        <form onSubmit={createPatient} className="bg-white p-3 rounded flex gap-2">
          <input required placeholder="Name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} className="border p-1" />
          <input placeholder="Age" value={form.age} onChange={(e) => setForm({ ...form, age: e.target.value })} className="border p-1 w-16" />
          <select value={form.gender} onChange={(e) => setForm({ ...form, gender: e.target.value })} className="border p-1">
            <option value="">Gender</option>
            <option>Male</option>
            <option>Female</option>
            <option>Other</option>
          </select>
          <button type="submit" disabled={creating} className="px-3 py-1 bg-green-600 text-white rounded">Create</button>
        </form>
      </div>
      {error ? <div className="mb-4 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div> : null}
      <div className="grid grid-cols-3 gap-4">
        {patients.map((p) => (
          <div key={p.id} className="bg-white p-4 rounded shadow">
            <h2 className="font-semibold">{p.name}</h2>
            <div className="text-sm text-gray-600 mt-2">
              <div>X-rays: {p.xrays}</div>
              <div>CT: {p.ct}</div>
              <div>MRI: {p.mri}</div>
              <div>Reports: {p.reports}</div>
            </div>
            <div className="mt-3">
              <div className="h-3 bg-gray-200 rounded">
                <div className="h-3 bg-green-500 rounded" style={{ width: `${p.completion}%` }} />
              </div>
              <div className="text-sm text-gray-500 mt-1">Completion: {p.completion}%</div>
            </div>
            <a className="text-sm text-blue-600 mt-2 inline-block" href={`/patients/${p.id}`}>View</a>
          </div>
        ))}
      </div>
    </div>
  );
}
