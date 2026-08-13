"use client";
import { API_BASE } from '../constants';
import { useState } from 'react';


export default function CtBmdPage() {
  const [ctFile, setCtFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      if (!ctFile) {
        throw new Error('Please upload a CT file.');
      }
      const fd = new FormData();
      fd.append('file', ctFile);
      const res = await fetch(`${API_BASE}/api/run-ct-bmd`, {
        method: 'POST',
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || data.error || 'Unknown error');
      }
      setResult(data);
    } catch (err:any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">CT Bone Density Estimation (DeepBone)</h1>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <label className="block">
          <span className="text-gray-700">Upload a CT file</span>
          <input
            type="file"
            accept=".nii,.nii.gz,.dcm,.dicom,.mhd,.nrrd,.png,.jpg,.jpeg"
            onChange={(e) => setCtFile(e.target.files?.[0] || null)}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            required
          />
        </label>
        <button
          type="submit"
          disabled={loading}
          className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition"
        >
          {loading ? 'Processing…' : 'Run Estimation'}
        </button>
      </form>
      <p className="mt-2 text-sm text-gray-500">This page uploads the file to the backend; it no longer depends on a local path.</p>
      {error && <p className="mt-4 text-red-600">Error: {error}</p>}
      {result && (
        <div className="mt-6 p-4 border rounded bg-gray-50">
          <h2 className="text-xl font-semibold mb-2">Result</h2>
          <pre className="whitespace-pre-wrap text-sm">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
