"use client";
import React, { useEffect, useState } from "react";
import UploadDropzone from "../../../components/UploadDropzone";

export default function PatientPage() {
  const [patient, setPatient] = useState<any>(null);
  const id = typeof window !== "undefined" ? window.location.pathname.split("/").pop() : null;

  useEffect(() => {
    if (!id) return;
    fetch(`http://localhost:8000/api/patients/${id}`)
      .then((r) => r.json())
      .then(setPatient)
      .catch(console.error);
  }, [id]);

  if (!patient) return <div className="p-6">Loading...</div>;

  const onReportUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    const form = e.target as HTMLFormElement;
    const fd = new FormData(form);
    const res = await fetch(`http://localhost:8000/api/patients/${id}/upload-report`, { method: "POST", body: fd });
    if (res.ok) alert("Report uploaded");
    else alert("Failed to upload report");
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-xl font-bold">{patient.name}</h1>
      <p>Age: {patient.age} Gender: {patient.gender}</p>

      <section className="mt-4">
        <h2 className="font-semibold">Upload Images</h2>
        <UploadDropzone patientId={id || ""} />
      </section>

      <section className="mt-4">
        <h2 className="font-semibold">Upload Report</h2>
        <form onSubmit={onReportUpload} className="flex flex-col gap-2">
          <input type="file" name="file" />
          <textarea name="text" placeholder="Paste report text" className="border p-2" />
          <button type="submit" className="px-3 py-1 bg-blue-600 text-white rounded">Upload</button>
        </form>
      </section>

    </div>
  );
}
