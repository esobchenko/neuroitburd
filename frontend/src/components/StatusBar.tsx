import type { Job } from "../api";

interface Props {
  job: Job | null;
  submitting: boolean;
  error: string | null;
}

/** Job status line with a progress bar, or an error message. */
export default function StatusBar({ job, submitting, error }: Props) {
  if (error) {
    return (
      <p role="alert" className="rounded-md border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-300">
        {error}
      </p>
    );
  }
  if (submitting) return <Line text="Uploading…" progress={null} />;
  if (!job) return null;

  switch (job.status) {
    case "queued":
      return <Line text="Queued, waiting for the GPU…" progress={null} />;
    case "processing":
      return <Line text={`Painting… ${Math.round(job.progress * 100)}%`} progress={job.progress} />;
    case "completed":
      return <Line text={`Done in ${duration(job)} · seed ${job.params.seed}`} progress={1} />;
    case "failed":
      return (
        <p role="alert" className="rounded-md border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-300">
          Generation failed: {job.error ?? "unknown error"}
        </p>
      );
  }
}

function Line({ text, progress }: { text: string; progress: number | null }) {
  return (
    <div className="flex flex-col gap-2" aria-live="polite">
      <span className="text-sm text-neutral-300">{text}</span>
      <div
        className="h-1 overflow-hidden rounded-full bg-neutral-800"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progress === null ? undefined : Math.round(progress * 100)}
      >
        {progress === null ? (
          <div className="h-full w-1/3 animate-pulse rounded-full bg-neutral-600" />
        ) : (
          <div
            className="h-full rounded-full bg-accent transition-[width] duration-500"
            style={{ width: `${Math.max(progress * 100, 2)}%` }}
          />
        )}
      </div>
    </div>
  );
}

function duration(job: Job): string {
  if (!job.started_at || !job.finished_at) return "—";
  const seconds = (Date.parse(job.finished_at) - Date.parse(job.started_at)) / 1000;
  return `${seconds.toFixed(1)} s`;
}
