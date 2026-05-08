import { createFileRoute } from "@tanstack/react-router";
import StoolNetDiagram from "@/components/StoolNetDiagram";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "StoolNetTriple — Triple Attention CNN Architecture" },
      {
        name: "description",
        content:
          "Interactive visualisation of the StoolNetTriple Triple Attention CNN: position, channel, and type attention branches feeding three multi-task classification heads.",
      },
    ],
  }),
});

function Index() {
  return <StoolNetDiagram />;
}
