import { createFileRoute } from "@tanstack/react-router";
import ResNetCVPipeline from "@/components/ResNetCVPipeline";

export const Route = createFileRoute("/resnet-cv")({
  component: ResNetCVPage,
  head: () => ({
    meta: [
      { title: "ResNet50 · 5-Fold CV Pipeline" },
      {
        name: "description",
        content:
          "Interactive visualisation of the ResNet50 5-fold cross-validation training and inference pipeline with TTA ×8 and K-fold ensemble.",
      },
    ],
  }),
});

function ResNetCVPage() {
  return <ResNetCVPipeline />;
}
