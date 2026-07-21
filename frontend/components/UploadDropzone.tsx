"use client";
import React, { useCallback, useState } from "react";

type Props = {
  patientId: string;
  onUploadComplete?: () => void;
};

export default function UploadDropzone({ patientId, onUploadComplete }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const onFiles = useCallback((selected: FileList | null) => {
    if (!selected) return;
    const arr = Array.from(selected);
    setFiles((s) => [...s, ...arr]);
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    onFiles(e.dataTransfer.files);
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const onUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    setStatusMsg(null);
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      const res = await fetch(`http://localhost:8000/api/patients/${patientId}/upload-image`, {
        method: "POST",
        body: fd,
      });
      if (res.ok) {
        setStatusMsg("Image uploaded & analyzed successfully!");
        setFiles([]);
        if (onUploadComplete) onUploadComplete();
      } else {
        setStatusMsg("Upload failed. Please try again.");
      }
    } catch (err) {
      setStatusMsg("Failed to connect to backend server.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 text-slate-100 shadow-xl">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        className="border-2 border-dashed border-sky-500/30 hover:border-sky-400 bg-slate-800/50 hover:bg-slate-800 rounded-lg p-6 text-center transition cursor-pointer"
      >
        <div className="mx-auto w-12 h-12 mb-3 rounded-full bg-sky-500/10 flex items-center justify-center text-sky-400">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 0115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>
        <p className="font-medium text-slate-200">Drag & drop X-ray images here</p>
        <p className="text-xs text-slate-400 mt-1">Supports PNG, JPG, DICOM (.dcm)</p>
        
        <label className="inline-block mt-3 px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold rounded-lg cursor-pointer transition shadow">
          Browse Files
          <input type="file" multiple accept="image/*,.dcm" onChange={(e) => onFiles(e.target.files)} className="hidden" />
        </label>
      </div>

      {files.length > 0 && (
        <div className="mt-4 space-y-2">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Selected Files ({files.length})</h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {files.map((f, i) => (
              <div key={i} className="flex items-center justify-between bg-slate-800 p-2.5 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 overflow-hidden">
                  {f.name.toLowerCase().endsWith(".dcm") ? (
                    <div className="w-8 h-8 bg-sky-950 text-sky-400 text-xs font-bold rounded flex items-center justify-center shrink-0">DCM</div>
                  ) : (
                    <img src={URL.createObjectURL(f)} className="w-8 h-8 rounded object-cover shrink-0" alt={f.name} />
                  )}
                  <span className="text-xs font-medium truncate text-slate-200">{f.name}</span>
                </div>
                <button onClick={() => removeFile(i)} className="text-slate-400 hover:text-red-400 p-1 text-xs">✕</button>
              </div>
            ))}
          </div>
          
          <button
            onClick={onUpload}
            disabled={uploading}
            className="w-full mt-3 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 text-white text-sm font-semibold rounded-lg transition shadow flex items-center justify-center gap-2"
          >
            {uploading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                </svg>
                Processing & Analyzing XAI...
              </>
            ) : (
              `Upload & Analyze (${files.length})`
            )}
          </button>
        </div>
      )}

      {statusMsg && (
        <div className={`mt-3 p-3 rounded-lg text-xs font-medium ${statusMsg.includes("failed") ? "bg-red-500/20 text-red-300 border border-red-500/30" : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"}`}>
          {statusMsg}
        </div>
      )}
    </div>
  );
}

