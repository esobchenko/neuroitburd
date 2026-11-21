import { useRef, useState, type DragEvent, type ReactNode } from "react";
import { LIMITS } from "../api";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
  children: ReactNode;
}

/** Click or drop target for one image file. Validation is left to the parent. */
export default function Dropzone({ onFile, disabled = false, children }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const pick = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onFile(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    if (!disabled) pick(e.dataTransfer.files);
  };

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload an image"
      aria-disabled={disabled}
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(e) => {
        if (!disabled && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          input.current?.click();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      className={`relative h-full w-full rounded-lg outline-none transition focus-visible:ring-2 focus-visible:ring-accent ${
        disabled ? "cursor-not-allowed" : "cursor-pointer"
      } ${over ? "ring-2 ring-accent" : ""}`}
    >
      {children}
      <input
        ref={input}
        type="file"
        accept={LIMITS.acceptedTypes.join(",")}
        className="hidden"
        onChange={(e) => {
          pick(e.target.files);
          e.target.value = ""; // allow re-selecting the same file
        }}
      />
    </div>
  );
}
