# KetabMind Kubernetes Manifests

This directory contains baseline manifests for deploying the KetabMind API, Qdrant vector database, and observability stack (Prometheus + Grafana) onto a Kubernetes cluster.

## Usage

1. Create the required ConfigMaps and Secrets referenced in the manifests (for example `ketabmind-api-config`, `ketabmind-api-secret`, and `grafana-admin`).
2. Apply the manifests in this directory:

   ```bash
   kubectl apply -f k8s/
   ```

3. Verify that the pods become ready:

   ```bash
   kubectl get pods
   ```

4. (Optional) Configure an Ingress controller and DNS entry for the FastAPI service host `api.ketabmind.example.com`.

## Production Considerations

- **Transport security:** Terminate TLS at the Ingress layer (e.g., cert-manager + Let's Encrypt) or use a service mesh (Istio/Linkerd) for mTLS between workloads.
- **Autoscaling:** Attach a HorizontalPodAutoscaler (HPA) to the API deployment based on CPU, memory, or custom metrics to handle load spikes.
- **Workload placement:** Use node selectors, affinity/anti-affinity, and taints/tolerations to isolate services onto dedicated nodes (e.g., separate compute, storage, and monitoring pools).
- **GPU scheduling:** If the API or Qdrant requires GPU acceleration, request GPU resources (e.g., `nvidia.com/gpu`) and ensure the cluster has the NVIDIA device plugin installed.
- **Stateful storage:** For production Qdrant deployments, choose a production-grade StorageClass (e.g., SSD-backed volumes) and configure backup/restore workflows.
- **Resource tuning:** Adjust resource requests/limits and retention policies based on real workload metrics collected through Prometheus/Grafana.
