import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const audio = formData.get("audio") as Blob;
    let language = formData.get("language") as string;
    
    if (!audio) {
      return NextResponse.json({ error: "Missing audio payload" }, { status: 400 });
    }

    // Force explicit MIME type because Next.js req.formData() sometimes parses it as application/octet-stream
    const audioBuffer = await audio.arrayBuffer();
    const cleanAudio = new File([audioBuffer], "audio.webm", { type: "audio/webm" });

    const sarvamFormData = new FormData();
    sarvamFormData.append("file", cleanAudio);
    sarvamFormData.append("model", "saaras:v3");
    sarvamFormData.append("with_timestamps", "false");
    sarvamFormData.append("with_disfluencies", "false");
    
    if (!language || language === "auto" || language === "unknown") {
      sarvamFormData.append("language_code", "unknown");
    } else {
      sarvamFormData.append("language_code", language);
    }

    const sarvamKey = process.env.SARVAM_API_KEY;
    if (!sarvamKey) {
      return NextResponse.json({ error: "SARVAM_API_KEY is not configured on Vercel" }, { status: 500 });
    }

    const sttReq = await fetch("https://api.sarvam.ai/speech-to-text", {
      method: "POST",
      headers: {
        "api-subscription-key": sarvamKey,
      },
      body: sarvamFormData,
    });

    if (!sttReq.ok) {
      const errText = await sttReq.text();
      console.error("Sarvam STT Error:", errText);
      try {
        const errJson = JSON.parse(errText);
        return NextResponse.json({ error: `Sarvam API error: ${errJson.error?.message || errText}` }, { status: 500 });
      } catch {
        return NextResponse.json({ error: `Sarvam API error: ${errText}` }, { status: 500 });
      }
    }

    const sttResult = await sttReq.json();
    return NextResponse.json({
      transcript: sttResult.transcript,
      language: sttResult.language_code
    });

  } catch (err: any) {
    console.error("Vercel Edge STT Error:", err);
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
