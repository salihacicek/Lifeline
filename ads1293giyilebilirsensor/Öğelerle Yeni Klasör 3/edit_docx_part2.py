from docx import Document
import sys

def process_file():
    input_file = "/Users/salihacicek/Desktop/GUNCEL_Part_B1.docx"
    output_file = "/Users/salihacicek/Desktop/GUNCEL_Part_B1_TAMAMLANDI.docx"

    try:
        doc = Document(input_file)
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    excellence_text = """Autonomous QA/QC Pipelines and Algorithmic Mitigation of Data Acquisition Delays
To address the inherent risks of data acquisition delays and heterogeneous image quality from multi-center clinical field environments, the methodology is strictly decoupled from manual, sequential data grading. An automated Quality Assurance and Quality Control (QA/QC) pipeline will be executed pre-inference. For raw B-mode ultrasound and SWE data, a lightweight Convolutional Neural Network (MobileNetV3-Small) will evaluate transducer coupling and acoustic shadowing, outputting a structural similarity index measure (SSIM). Scans scoring an SSIM <0.75 are autonomously rejected, preventing the contamination of the downstream predictive pipeline.

Furthermore, any longitudinal data acquisition delays or incomplete multimodal matrices will not stall the machine learning workflow. To decouple model training from clinical delays, the architecture employs a sparsity-aware XGBoost framework, which natively handles missing tensors by learning default directional splits. In cases of severe multimodal data sparsity (e.g., missing MRI follow-ups), missing biomechanical and morphological vectors will be imputed using a pre-trained Generative Adversarial Network (GAN) architecture specifically optimized for tabular-radiomic imputation, ensuring the Elastic Net and XGBoost classifiers continuously receive dense, orthogonal matrices for ongoing hyperparameter tuning without waiting for full cohort completion."""

    impact_text = """Intellectual Property (IP) Protection and Software as a Medical Device (SaMD) Commercialization Architecture
The 'Hamstring Risk Card' transcends a theoretical framework by being architected as a deployable, cloud-native Software as a Medical Device (SaMD). To strictly protect the project's intellectual property and prevent reverse-engineering of the predictive models, the core algorithmic assets—specifically the U-Net spatial weights and the XGBoost decision tree ensembles—will be legally and architecturally maintained as Trade Secrets. These models will be hosted in isolated, hardware-encrypted cloud enclaves (e.g., AWS Nitro Enclaves).

End-users (elite clubs) will not have access to the raw source code or model weights. Instead, integration will occur exclusively via containerized, rate-limited RESTful API endpoints. This robust backend separation inherently protects the core IP while facilitating a highly scalable Business-to-Business (B2B) Software as a Service (SaaS) commercialization route. Provisional patents (Invention Disclosures) will be filed specifically covering the proprietary cross-modality data fusion vectors (combining SWE elastograms with MRI meshes) prior to any open-access publication. The front-end dashboard will be licensed to elite clubs via secure OAuth2 authentication protocols, ensuring GDPR-compliant data transmission and creating a direct revenue-generation pipeline post-fellowship."""

    implementation_text = """Agile MLOps Integration and High-Resolution Predictive Milestones
The assertion that the integration timeframe is constrained is mitigated by replacing traditional waterfall software development with an Agile Machine Learning Operations (MLOps) CI/CD (Continuous Integration / Continuous Deployment) pipeline. Model training is not deferred until the completion of prospective data collection; rather, the baseline algorithms will be pre-trained on the retrospective SIRP-600 dataset and iteratively updated via dynamic retraining triggers.

To ensure granular tracking of the data science and computer vision pipelines, the following highly specific technical milestones are integrated into the 36-month timeline:
- M6 (V1 Model Freeze): Initial hyperparameter optimization (via GridSearchCV) and feature selection (Boruta algorithm) completed on the retrospective SIRP-600 dataset. Baseline XGBoost AUC established.
- M12 (Cross-Modality Pipeline Deployed): Autonomous spatial registration scripts mapping portable ultrasound coordinates to MRI macroscopic meshes validated (SSIM >0.85).
- M18 (Prospective Ingestion Checkpoint): First batch of prospective multimodal vectors ingested via the QA/QC API; concept drift metrics evaluated to trigger automatic model weight recalibration.
- M24 (External Validation Complete): Out-of-Fold (OOF) cross-validation on the secondary geographic cohort (Maynooth) finalised, verifying model robustness against demographic variance.
- M36 (SaMD API Hardening): Final XGBoost and U-Net RESTful endpoints stress-tested, load-balanced, and secured for commercial clinical deployment across elite clubs."""

    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip().lower()
        if "1.2\tsoundness of the proposed methodology" in text:
            p.insert_paragraph_before("\n\n" + excellence_text)
        elif "2.2\tsuitability and quality of the measures to maximise expected outcomes" in text:
            p.insert_paragraph_before("\n\n" + impact_text)
        elif "3.1\tquality and effectiveness of the work plan" in text:
            p.insert_paragraph_before("\n\n" + implementation_text)

    try:
        doc.save(output_file)
        print(f"File saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    process_file()
