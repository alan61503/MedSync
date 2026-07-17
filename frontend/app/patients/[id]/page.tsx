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

  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [inference, setInference] = useState<any>(null);

  const fetchInference = async (imagePath: string) => {
    // imagePath is a url under /uploads/...; call backend run-inference endpoint
    try {
      const res = await fetch(`http://localhost:8000/api/run-inference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: imagePath }),
      });
      const data = await res.json();
      setInference(data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-xl font-bold">{patient.name}</h1>
      <p>Age: {patient.age} Gender: {patient.gender}</p>

      <section className="mt-4">
        <h2 className="font-semibold">Upload Images</h2>
        <UploadDropzone patientId={id || ""} />
        <div className="mt-4">
          <h3 className="font-medium">Uploaded Images</h3>
          <div>
            {patient.images.map((img: any) => (
              <div key={img.id} className="flex items-center gap-3 py-2">
                <div className="w-24 h-24 bg-gray-100 flex items-center justify-center">
                  <img src={img.file_path.replace(/^\//, "http://localhost:8000/") } alt={img.filename} className="object-contain max-h-24" />
                </div>
                <div>
                  <div className="font-semibold">{img.filename}</div>
                  <div className="text-sm text-gray-500">Modality: {img.modality}</div>
                  <div>
                    <button className="text-sm text-blue-600" onClick={() => { setSelectedImage(img.file_path); fetchInference(img.file_path.replace(/^\//, "http://localhost:8000/")); }}>Run Inference</button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mt-4">
        <h2 className="font-semibold">Upload Report</h2>
        <form onSubmit={onReportUpload} className="flex flex-col gap-2">
          <input type="file" name="file" />
          <textarea name="text" placeholder="Paste report text" className="border p-2" />
          <button type="submit" className="px-3 py-1 bg-blue-600 text-white rounded">Upload</button>
        </form>
      </section>

      {inference && (
        <section className="mt-6 bg-white p-4 rounded shadow">
          <h3 className="font-semibold">Inference Results</h3>
          {inference.error && <div className="text-red-600">Error: {inference.error}</div>}
          <div className="grid grid-cols-2 gap-4 mt-3">
            <div>
              <h4 className="font-medium">Top Predictions</h4>
              <ul>
                {Object.entries(inference.predictions || {}).sort((a: any, b: any) => (b[1] as number) - (a[1] as number)).slice(0,5).map(([k, v]: any) => (
                  <li key={k} className="py-1">
                    <div className="flex items-center justify-between">
                      <div className="text-sm">{k.replace(/_/g, ' ')}</div>
                      <div className="text-xs text-gray-600">{(v as number).toFixed(2)}</div>
                    </div>
                    <div className="h-2 bg-gray-200 rounded mt-1">
                      <div className="h-2 bg-indigo-500 rounded" style={{ width: `${Math.min(100, (v as number) * 100)}%` }} />
                    </div>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="font-medium">Heatmap</h4>
              {inference.heatmap_path ? (
                <img src={(inference.heatmap_path as string).startsWith('/uploads') ? `http://localhost:8000${inference.heatmap_path}` : inference.heatmap_path} alt="heatmap" className="max-h-96 object-contain" />
              ) : (
                <div className="text-sm text-gray-500">No heatmap available</div>
              )}
            </div>
          </div>
        </section>
      )}

    </div>
  );
}
