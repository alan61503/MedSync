export const loadFirebase = async () => {
  const [{ initializeApp, getApps, getApp }, { getAnalytics, isSupported }] =
    await Promise.all([
      import('firebase/app'),
      import('firebase/analytics'),
    ]);
  const firebaseConfig = {
    apiKey: "AIzaSyCOFn5nfG3EMMS7ku_wpjn5QJG6kUaWbD8",
    authDomain: "medsync-619ed.firebaseapp.com",
    projectId: "medsync-619ed",
    storageBucket: "medsync-619ed.firebasestorage.app",
    messagingSenderId: "856570035998",
    appId: "1:856570035998:web:05db624fee08efc9c4530f",
    measurementId: "G-R3RFSLQ4VG",
  };
  const app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
  const analytics = (await isSupported()) ? getAnalytics(app) : null;
  return { app, analytics };
};

