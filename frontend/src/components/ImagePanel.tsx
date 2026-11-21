import type { ReactNode } from "react";

interface Props {
  title: string;
  src: string | null;
  alt: string;
  placeholder: ReactNode;
  footer?: ReactNode;
  dimmed?: boolean;
}

/** A titled frame showing an image, or a placeholder when there is none. */
export default function ImagePanel({ title, src, alt, placeholder, footer, dimmed = false }: Props) {
  return (
    <figure className="flex min-w-0 flex-col gap-2">
      <figcaption className="flex items-center justify-between text-xs uppercase tracking-widest text-neutral-500">
        {title}
      </figcaption>
      <div className="flex aspect-[4/3] items-center justify-center overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900">
        {src ? (
          <img
            src={src}
            alt={alt}
            className={`h-full w-full object-contain transition-opacity ${dimmed ? "opacity-40" : ""}`}
          />
        ) : (
          <div className="p-6 text-center text-sm text-neutral-500">{placeholder}</div>
        )}
      </div>
      {footer && <div className="min-h-8 text-sm">{footer}</div>}
    </figure>
  );
}
