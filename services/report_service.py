def assemble_final_report(*, diagnosis, confidence, explanation, recommendations, heatmap, verification_status):
    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "explanation": explanation,
        "recommendations": recommendations,
        "heatmap": heatmap,
        "verification_status": verification_status,
    }
