import type { ReactNode } from "react";
import { LIMITS, type GenerateRequest } from "../api";

interface Props {
  values: GenerateRequest;
  onChange: (values: GenerateRequest) => void;
  disabled: boolean;
}

/** Basic controls (strength, steps, seed) plus a collapsible advanced section. */
export default function Controls({ values, onChange, disabled }: Props) {
  const set = <K extends keyof GenerateRequest>(key: K, value: GenerateRequest[K]) =>
    onChange({ ...values, [key]: value });

  const effectiveSteps = Math.floor(values.steps * values.strength);

  return (
    <fieldset disabled={disabled} className="flex flex-col gap-5 disabled:opacity-60">
      <Slider
        label="Strength"
        hint="How far to move away from the original"
        value={values.strength}
        min={0.05}
        max={1}
        step={0.05}
        format={(v) => v.toFixed(2)}
        onChange={(v) => set("strength", v)}
      />
      <Slider
        label="Steps"
        hint={`${effectiveSteps} denoising steps will run (steps × strength)`}
        value={values.steps}
        min={1}
        max={100}
        step={1}
        format={(v) => String(v)}
        onChange={(v) => set("steps", v)}
      />

      <Field label="Seed" hint="Leave empty for a random seed">
        <div className="flex gap-2">
          <input
            type="number"
            inputMode="numeric"
            min={0}
            max={LIMITS.maxSeed}
            placeholder="random"
            value={values.seed}
            onChange={(e) => set("seed", e.target.value)}
            className={inputClass}
          />
          <button
            type="button"
            onClick={() => set("seed", String(Math.floor(Math.random() * LIMITS.maxSeed)))}
            className="shrink-0 rounded-md border border-neutral-700 px-3 text-sm text-neutral-300 hover:border-neutral-500 hover:text-white"
            title="Pick a random seed"
          >
            Randomize
          </button>
        </div>
      </Field>

      <details className="group rounded-md border border-neutral-800 px-3 py-2">
        <summary className="cursor-pointer select-none text-sm text-neutral-400 hover:text-neutral-200">
          Advanced
        </summary>
        <div className="mt-4 flex flex-col gap-5 pb-2">
          <Slider
            label="Guidance"
            hint="How strictly to follow the prompt"
            value={values.guidance}
            min={0}
            max={20}
            step={0.5}
            format={(v) => v.toFixed(1)}
            onChange={(v) => set("guidance", v)}
          />
          <Field label="Prompt" hint="Empty uses the default style prompt">
            <textarea
              rows={3}
              maxLength={LIMITS.maxPromptChars}
              placeholder="An expressive postmodern oil painting…"
              value={values.prompt}
              onChange={(e) => set("prompt", e.target.value)}
              className={`${inputClass} resize-y`}
            />
          </Field>
          <Field label="Negative prompt" hint="Empty uses the default">
            <textarea
              rows={2}
              maxLength={LIMITS.maxPromptChars}
              placeholder="photo, photorealistic, blurry…"
              value={values.negativePrompt}
              onChange={(e) => set("negativePrompt", e.target.value)}
              className={`${inputClass} resize-y`}
            />
          </Field>
        </div>
      </details>
    </fieldset>
  );
}

const inputClass =
  "w-full min-w-0 rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-600 focus:border-accent focus:outline-none";

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm text-neutral-300">{label}</span>
      {children}
      {hint && <span className="text-xs text-neutral-500">{hint}</span>}
    </label>
  );
}

interface SliderProps {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  onChange: (value: number) => void;
}

function Slider({ label, hint, value, min, max, step, format, onChange }: SliderProps) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="flex justify-between text-sm">
        <span className="text-neutral-300">{label}</span>
        <span className="tabular-nums text-neutral-400">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <span className="text-xs text-neutral-500">{hint}</span>}
    </label>
  );
}
