import { useEffect, useRef, useState } from "react";
import {
  checkFile,
  getHealth,
  isAbort,
  pollJob,
  submitJob,
  type GenerateRequest,
  type Health,
  type Job,
} from "./api";
import Controls from "./components/Controls";
import Dropzone from "./components/Dropzone";
import ImagePanel from "./components/ImagePanel";
import StatusBar from "./components/StatusBar";

const DEFAULTS: GenerateRequest = {
  strength: 0.65,
  steps: 30,
  guidance: 7,
  seed: "",
  prompt: "",
  negativePrompt: "",
};

export default function App() {
  const health = useHealth();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [values, setValues] = useState<GenerateRequest>(DEFAULTS);
  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  const running = submitting || job?.status === "queued" || job?.status === "processing";
  const modelFailed = health.data?.status === "error";
  const canGenerate = file !== null && !running && !modelFailed && !health.offline;

  // Release the object URL of the previous preview, and stop polling on unmount.
  useEffect(() => () => void (previewUrl && URL.revokeObjectURL(previewUrl)), [previewUrl]);
  useEffect(() => () => abort.current?.abort(), []);

  const chooseFile = (next: File) => {
    const problem = checkFile(next);
    if (problem) {
      setError(problem);
      return;
    }
    setFile(next);
    setPreviewUrl(URL.createObjectURL(next));
    setJob(null);
    setError(null);
  };

  const generate = async () => {
    if (!file) return;
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setError(null);
    setJob(null);
    setSubmitting(true);
    try {
      const created = await submitJob(file, values, controller.signal);
      setSubmitting(false);
      setJob(created);
      await pollJob(created.job_id, setJob, controller.signal);
    } catch (err) {
      if (isAbort(err)) return;
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const result = job?.status === "completed" ? job.output_url : null;

  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:py-10">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <div className="flex items-baseline gap-3">
          <h1 className="text-2xl font-semibold tracking-tight text-white">Neuroitburd</h1>
          <span className="text-sm text-neutral-500">photo → expressive painting</span>
        </div>
        <HealthBadge health={health.data} offline={health.offline} />
      </header>

      {modelFailed && health.data?.model_error && (
        <p role="alert" className="rounded-md border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-300">
          The model failed to load: {health.data.model_error}
        </p>
      )}

      <main className="grid flex-1 gap-8 lg:grid-cols-[20rem_1fr]">
        <section className="order-2 flex flex-col gap-6 lg:order-1">
          <Controls values={values} onChange={setValues} disabled={running} />
          <button
            type="button"
            onClick={generate}
            disabled={!canGenerate}
            className="rounded-md bg-accent px-4 py-2.5 font-medium text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:bg-neutral-800 disabled:text-neutral-500"
          >
            {running ? "Generating…" : "Generate"}
          </button>
          <StatusBar job={job} submitting={submitting} error={error} />
          <p className="text-xs leading-relaxed text-neutral-600">
            Style approximation — not a Roitburd-trained model. Inspired by the expressive,
            postmodern painting of Oleksandr Roitburd.
          </p>
        </section>

        <section className="order-1 grid content-start gap-6 md:grid-cols-2 lg:order-2">
          <Dropzone onFile={chooseFile} disabled={running}>
            <ImagePanel
              title="Original"
              src={previewUrl}
              alt="Uploaded original"
              placeholder={
                <>
                  <span className="block text-neutral-300">Drop an image here</span>
                  <span className="mt-1 block">or click to choose · JPEG, PNG, WebP · up to 20 MB</span>
                </>
              }
              footer={
                file && (
                  <span className="text-neutral-500">
                    {file.name} · {formatSize(file.size)}
                    {!running && " · click or drop to replace"}
                  </span>
                )
              }
            />
          </Dropzone>
          <ImagePanel
            title="Result"
            src={result}
            alt="Stylized result"
            placeholder={running ? "Painting…" : "The stylized image will appear here"}
            footer={
              result &&
              job && (
                <div className="flex flex-wrap items-center gap-3">
                  <a
                    href={result}
                    download={`neuroitburd-${job.params.seed}.png`}
                    className="rounded-md border border-neutral-700 px-3 py-1.5 text-neutral-200 hover:border-neutral-500 hover:text-white"
                  >
                    Download PNG
                  </a>
                  <button
                    type="button"
                    onClick={() => setValues({ ...values, seed: String(job.params.seed) })}
                    className="text-neutral-500 underline-offset-4 hover:text-neutral-300 hover:underline"
                  >
                    Reuse seed {job.params.seed}
                  </button>
                </div>
              )
            }
          />
        </section>
      </main>
    </div>
  );
}

/** Polls /api/health: every 2 s while the model loads, every 5 s while the backend is down. */
function useHealth(): { data: Health | null; offline: boolean } {
  const [data, setData] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const check = async () => {
      try {
        const health = await getHealth(controller.signal);
        setData(health);
        setOffline(false);
        if (health.status === "starting") timer = setTimeout(check, 2000);
      } catch (err) {
        if (isAbort(err)) return;
        setOffline(true);
        timer = setTimeout(check, 5000);
      }
    };
    void check();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  return { data, offline };
}

function HealthBadge({ health, offline }: { health: Health | null; offline: boolean }) {
  let text = "Connecting…";
  let dot = "bg-neutral-500";
  if (offline) {
    text = "Backend offline";
    dot = "bg-red-500";
  } else if (health?.status === "starting") {
    text = "Loading model…";
    dot = "bg-amber-400 animate-pulse";
  } else if (health?.status === "error") {
    text = "Model error";
    dot = "bg-red-500";
  } else if (health?.status === "ok") {
    text = health.gpu ? health.gpu.name : `Ready · ${health.device ?? "cpu"}`;
    dot = "bg-emerald-500";
  }
  return (
    <span className="flex items-center gap-2 text-xs text-neutral-400" title={health?.model_id}>
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      {text}
    </span>
  );
}

function formatSize(bytes: number): string {
  return bytes < 1024 * 1024
    ? `${Math.max(1, Math.round(bytes / 1024))} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
