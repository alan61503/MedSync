"use client";
import React, { useCallback, useState } from "react";

type Props = {
  patientId: string;
};

export default function UploadDropzone({ patientId }: Props) {
  const [files, setFiles] = useState<File[]>([]);

  const onFiles = useCallback((selected: FileList | null) => {
    if (!selected) return;
    const arr = Array.from(selected);
    setFiles((s) => [...s, ...arr]);
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    onFiles(e.dataTransfer.files);
  };

  const onUpload = async () => {
    if (!files.length) return;
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const res = await fetch(`http://localhost:8000/api/patients/${patientId}/upload-image`, {
      method: "POST",
      body: fd,
    });
    if (res.ok) {
      alert("Uploaded");
      setFiles([]);
    } else {
      alert("Upload failed");
    }
  };

  return (
    <div>
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        className="p-4 border-2 border-dashed rounded bg-white"
      >
        <p className="text-sm">Drag & drop files here, or</p>
        <input type="file" multiple onChange={(e) => onFiles(e.target.files)} />
      </div>

      <div className="mt-2">
        {files.map((f, i) => (
          <div key={i} className="flex items-center gap-2 py-1">
            {f.name.toLowerCase().endsWith(".dcm") ? (
              <div className="w-10 h-10 bg-gray-100 flex items-center justify-center">DCM</div>
            ) : (
              <img src={URL.createObjectURL(f)} className="w-10 h-10 object-cover" alt={f.name} />
            )}
            <div className="text-sm">{f.name}</div>
          </div>
        ))}
      </div>

      <div className="mt-2">
        <button onClick={onUpload} className="px-3 py-1 bg-blue-600 text-white rounded">Upload</button>
      </div>
    </div>
  );
}
