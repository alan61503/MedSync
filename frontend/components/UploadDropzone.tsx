"use client";

import React, { useCallback, useState } from "react";

type Props = {
  patientId: string;
  onUploadComplete?: () => void;
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
      const res = await fetch(`${API_BASE}/api/patients/${patientId}/upload-image`, {
        method: "POST",
        body: fd,
      });
      if (res.ok) {
        setStatusMsg("Upload successful & XAI generated!");
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
    <div className="bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm text-slate-900">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        className="border-2 border-dashed border-slate-200 hover:border-indigo-400 bg-slate-50/50 hover:bg-slate-50 rounded-xl p-5 text-center transition cursor-pointer"
      >
        <div className="mx-auto w-10 h-10 mb-2 rounded-full bg-indigo-50 flex items-center justify-center text-indigo-600">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 0115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>
        <p className="font-semibold text-slate-800 text-xs">Drop X-ray image here</p>
        <p className="text-[11px] text-slate-500 mt-0.5">Supports PNG, JPG, DICOM (.dcm)</p>

        <label className="inline-block mt-3 px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold rounded-lg cursor-pointer transition shadow-sm">
          Browse Files
          <input
            type="file"
            multiple
            accept="image/*,.dcm"
            onChange={(e) => onFiles(e.target.files)}
            className="hidden"
          />
        </label>
      </div>

      {files.length > 0 && (
        <div className="mt-4 space-y-2">
          <h4 className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Files Ready ({files.length})</h4>
          <div className="space-y-1.5 max-h-32 overflow-y-auto pr-1">
            {files.map((f, i) => (
              <div key={i} className="flex items-center justify-between bg-slate-50 p-2 rounded-lg border border-slate-200/60">
                <div className="flex items-center gap-2 overflow-hidden">
                  {f.name.toLowerCase().endsWith(".dcm") ? (
                    <div className="w-7 h-7 bg-indigo-100 text-indigo-700 text-[10px] font-bold rounded flex items-center justify-center shrink-0">DCM</div>
                  ) : (
                    <img src={URL.createObjectURL(f)} className="w-7 h-7 rounded object-cover shrink-0" alt={f.name} />
                  )}
                  <span className="text-xs font-medium truncate text-slate-700">{f.name}</span>
                </div>
                <button onClick={() => removeFile(i)} className="text-slate-400 hover:text-red-500 p-1 text-xs">✕</button>
              </div>
            ))}
          </div>

          <button
            onClick={onUpload}
            disabled={uploading}
            className="w-full mt-3 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white text-xs font-bold rounded-lg transition shadow flex items-center justify-center gap-2"
          >
            {uploading ? (
              <>
                <svg className="animate-spin h-3.5 w-3.5 text-white" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                </svg>
                Processing AI & XAI...
              </>
            ) : (
              `Upload & Analyze (${files.length})`
            )}
          </button>
        </div>
      )}

      {statusMsg && (
        <div className={`mt-3 p-2.5 rounded-lg text-xs font-medium ${statusMsg.includes("failed") ? "bg-red-50 text-red-700 border border-red-200" : "bg-emerald-50 text-emerald-700 border border-emerald-200"}`}>
          {statusMsg}
        </div>
      )}
    </div>
  );
}
