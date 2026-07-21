import { initializeApp, getApps, getApp, FirebaseApp } from "firebase/app";
import { getAnalytics, isSupported, Analytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyCOFn5nfG3EMMS7ku_wpjn5QJG6kUaWbD8",
  authDomain: "medsync-619ed.firebaseapp.com",
  projectId: "medsync-619ed",
  storageBucket: "medsync-619ed.firebasestorage.app",
  messagingSenderId: "856570035998",
  appId: "1:856570035998:web:05db624fee08efc9c4530f",
  measurementId: "G-R3RFSLQ4VG"
};

let app: FirebaseApp | null = null;
let analytics: Analytics | null = null;

export const initFirebase = () => {
  if (typeof window === "undefined") return { app: null, analytics: null };
  if (!app) {
    app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
    isSupported().then((supported) => {
      if (supported && app) {
        analytics = getAnalytics(app);
      }
    }).catch(() => {});
  }
  return { app, analytics };
};

export { app, analytics };
