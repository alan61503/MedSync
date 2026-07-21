import "../styles/globals.css";

export const metadata = {
  title: "MedSync | Osteoporosis AI & XAI Diagnosis",
  description: "Medical Image Analysis & Explainable AI Saliency Heatmaps",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">
        <header className="bg-slate-900/90 backdrop-blur-md border-b border-slate-800 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-6 py-3.5 flex items-center justify-between">
            <a href="/" className="flex items-center gap-3 group">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-500 to-emerald-400 flex items-center justify-center font-black text-slate-950 text-base shadow-lg shadow-sky-500/20 group-hover:scale-105 transition">
                M
              </div>
              <div>
                <div className="font-bold text-white tracking-tight text-base flex items-center gap-2">
                  MedSync <span className="text-[10px] font-extrabold uppercase bg-sky-500/20 text-sky-400 px-2 py-0.5 rounded border border-sky-500/30">XAI AI</span>
                </div>
                <div className="text-[10px] text-slate-400 tracking-wide">Osteoporosis Diagnostic Research</div>
              </div>
            </a>

            <nav className="flex items-center gap-4 text-xs font-semibold text-slate-400">
              <a href="/" className="hover:text-white transition px-3 py-1.5 rounded-lg hover:bg-slate-800">
                Patients Directory
              </a>
              <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[11px]">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                Backend API Connected
              </div>
            </nav>
          </div>
        </header>

        <main className="flex-1 max-w-7xl w-full mx-auto p-6">{children}</main>

        <footer className="border-t border-slate-900 bg-slate-950 py-4 text-center text-xs text-slate-500">
          MedSync Medical Research Platform &bull; Osteoporosis & Grad-CAM XAI Explainability Engine
        </footer>
      </body>
    </html>
  );
}

