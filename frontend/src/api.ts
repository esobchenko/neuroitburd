// Typed client for the Neuroitburd backend (see backend/neuroitburd/api.py).

export type JobStatus = "queued" | "processing" | "completed" | "failed";

export interface GenerationParams {
  strength: number;
  steps: number;
  guidance: number;
  seed: number | null;
  prompt: string | null;
  negative_prompt: string | null;
}

export interface Job {
  job_id: string;
  status: JobStatus;
  progress: number; // 0 to 1
  error: string | null;
  output_url: string | null;
  params: GenerationParams;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface Health {
  status: "starting" | "ok" | "error";
  model_status: "not_started" | "loading" | "ready" | "error";
  model_error: string | null;
  model_id: string;
  device: string | null;
  lora: { name: string; weight: number | null } | null;
  gpu: { name: string; total_mb: number; free_mb: number } | null;
  queue_size: number;
}

/** Form values as typed by the user; blank strings mean "use the server default". */
export interface GenerateRequest {
  strength: number;
  steps: number;
  guidance: number;
  seed: string;
  prompt: string;
  negativePrompt: string;
}

/** Same limits as the backend; the server stays the authority. */
export const LIMITS = {
  maxUploadMb: 20,
  acceptedTypes: ["image/jpeg", "image/png", "image/webp"],
  maxSeed: 2 ** 32 - 1,
  maxPromptChars: 500,
} as const;

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("Cannot reach the backend. Is the server running?", 0);
  }
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body && typeof body.detail === "string"
        ? body.detail
        : `Request failed (HTTP ${res.status})`;
    throw new ApiError(detail, res.status);
  }
  return body as T;
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return request<Health>("/api/health", { signal });
}

export function getJob(jobId: string, signal?: AbortSignal): Promise<Job> {
  return request<Job>(`/api/jobs/${encodeURIComponent(jobId)}`, { signal });
}

export function submitJob(file: File, values: GenerateRequest, signal?: AbortSignal): Promise<Job> {
  const form = new FormData();
  form.append("image", file);
  form.append("strength", String(values.strength));
  form.append("steps", String(values.steps));
  form.append("guidance", String(values.guidance));
  form.append("seed", values.seed.trim());
  form.append("prompt", values.prompt.trim());
  form.append("negative_prompt", values.negativePrompt.trim());
  return request<Job>("/api/generate", { method: "POST", body: form, signal });
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });

/** Poll a job until it completes or fails, reporting every update. */
export async function pollJob(
  jobId: string,
  onUpdate: (job: Job) => void,
  signal?: AbortSignal,
  intervalMs = 1000,
): Promise<Job> {
  for (;;) {
    const job = await getJob(jobId, signal);
    onUpdate(job);
    if (job.status === "completed" || job.status === "failed") return job;
    await sleep(intervalMs, signal);
  }
}

/** Client-side check before uploading; returns an error message or null. */
export function checkFile(file: File): string | null {
  if (!(LIMITS.acceptedTypes as readonly string[]).includes(file.type)) {
    return "Unsupported file type. Use a JPEG, PNG or WebP image.";
  }
  if (file.size > LIMITS.maxUploadMb * 1024 * 1024) {
    return `File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). The limit is ${LIMITS.maxUploadMb} MB.`;
  }
  return null;
}

export function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}
