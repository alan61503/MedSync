"use client";
import { useEffect } from "react";
import { loadFirebase } from "../lib/firebase";

export default function FirebaseInit() {
  useEffect(() => {
    // Load Firebase only in the browser.
    loadFirebase().catch(() => {
      console.warn("Firebase failed to initialize – ignored in dev.");
    });
  }, []);

  return null;
}
