import "../styles/globals.css";
import FirebaseInit from "../components/FirebaseInit";
import Link from "next/link";

export const metadata = {
  title: "MedSync | Osteoporosis Diagnostics & XAI",
  description: "Minimalist Medical X-Ray Inference & Explainable AI Visualizer",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="bg-slate-50 text-slate-900 min-h-screen flex flex-col font-['Plus_Jakarta_Sans',sans-serif] antialiased">
        <FirebaseInit />

        {/* Minimalist Professional Light Header */}
        <header className="bg-white/90 backdrop-blur-md border-b border-slate-200/80 sticky top-0 z-50">
          <div className="max-w-6xl mx-auto px-6 py-3.5 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center font-bold text-white text-lg shadow-md shadow-indigo-500/20 group-hover:scale-105 transition">
                ┼
              </div>
              <div>
                <div className="font-extrabold text-slate-900 text-lg tracking-tight flex items-center gap-2">
                  MedSync
                  <span className="text-[10px] font-bold uppercase tracking-wider bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full border border-indigo-200">
                    XAI Studio
                  </span>
                </div>
              </div>
            </Link>

            <nav className="flex items-center gap-4 text-xs font-semibold text-slate-600">
              <Link href="/" className="hover:text-indigo-600 transition px-3 py-1.5 rounded-lg hover:bg-slate-100">
                Patients Directory
              </Link>
              <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-[11px] font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                AI Model Active
              </div>
            </nav>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 max-w-6xl w-full mx-auto p-6">{children}</main>

        {/* Footer */}
        <footer className="border-t border-slate-200/80 bg-white py-4 text-center text-xs text-slate-500">
          MedSync Medical AI Platform &bull; Osteoporosis & Grad-CAM XAI Engine
        </footer>
      </body>
    </html>
  );
}
