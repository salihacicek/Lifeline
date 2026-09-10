from docx import Document
from docx.shared import RGBColor
import sys

def add_red_header(p, header_text):
    run = p.add_run("\n\n" + header_text + "\n")
    run.font.color.rgb = RGBColor(255, 0, 0)
    run.bold = True

def add_colored_injection_before(p, header_text, body_text):
    new_p = p.insert_paragraph_before()
    run1 = new_p.add_run(header_text + "\n")
    run1.font.color.rgb = RGBColor(255, 0, 0)
    run1.bold = True
    new_p.add_run(body_text + "\n")

def process_file():
    input_file = "/Users/salihacicek/Desktop/(Part B1) (HE MSCA PF) NEW.docx"
    output_file = "/Users/salihacicek/Desktop/GUNCEL_Part_B1_RENKLI.docx"

    try:
        doc = Document(input_file)
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    # Texts
    wp1_head = 'EKLENEN PART (Hakemlerin "Yöntemler jenerik" eleştirisini çürütmek için, Görüntü İşleme ve veri harmonizasyonu detayları eklenmiştir)'
    wp1_body = """The integration of macroscopic 3D MRI morphology (Springbok Analytics) with localized, 2D micro-architectural data (B-mode ultrasound and SWE) presents a significant multimodal fusion challenge. Since direct voxel-to-pixel registration is geometrically intractable, we will implement a feature-based spatial alignment protocol. Both MRI and ultrasound/SWE data will be mapped to a standardized 1D anatomical coordinate system, normalized by the relative distance from the ischial tuberosity to the fibular head. This ensures precise intra- and inter-rater reliability (targeting ICC >= 0.85 and CV <= 10% for repeated measures).

Furthermore, raw sonographic and elastographic data from the field are inherently corrupted by speckle noise, acoustic shadowing, and transducer-pressure artifacts. To prevent error propagation into the downstream modelling pipeline, an autonomous, headless preprocessing script will be deployed. For B-mode US, anisotropic diffusion filtering (Perona-Malik algorithm) will be applied to iteratively smooth intra-fascicular speckle while strictly preserving the high-frequency edges of the deep and superficial aponeuroses. For SWE, elastograms will be passed through a normalized cross-correlation mask; spatial regions exhibiting a shear-wave Signal-to-Noise Ratio (SNR) of <10 dB will be autonomously flagged as invalid and excluded from the computation of the Young’s modulus, ensuring absolute data fidelity prior to predictive vectorization."""

    wp3_head = 'EKLENEN PART (Hakemlerin "Yöntemler jenerik" eleştirisini çürütmek için, XGBoost vektörizasyon ve boyut küçültme stratejileri eklenmiştir)'
    wp3_body = """Tree-based algorithms (Gradient-boosted trees/XGBoost) and penalised linear regressors (Elastic Net) cannot directly ingest raw 3D meshes or 2D heatmaps. Consequently, an intermediate deterministic feature extraction (vectorization) pipeline will be developed. Rather than reducing complex SWE elastograms to a single 'mean stiffness' scalar, we will extract first-order statistical moments (variance, skewness, kurtosis) and second-order textural features using Gray-Level Co-occurrence Matrices (GLCM) to mathematically quantify heterogeneous stiffness patterns within the Biceps Femoris. Similarly, 3D MRI outputs will be vectorized into spatial gradients of intramuscular adipose tissue (IMAT) and cross-sectional area (CSA) asymmetry indices.

This rigorous feature engineering will yield a high-dimensional vector space. Given the cohort size (n=300 recruited athletes), feeding this raw matrix into the XGBoost classifier would inevitably induce the curse of dimensionality and severe overfitting. To strictly enforce model generalizability across temporal and geographic validation cohorts, a two-stage dimensionality reduction architecture will be implemented. First, a collinearity filter will eliminate redundant parameters (dropping variables with a Pearson correlation coefficient |r| > 0.85). Subsequently, the Boruta algorithm will isolate statistically significant predictive features, independent of the background noise. This will condense the feature matrix to an optimal subset, maintaining a minimum Events-Per-Variable (EPV) ratio of 10:1. The final model will target an AUC/C-index >= 0.70 and Brier score <= 0.20, with full uncertainty reported via nested internal validation."""

    exc_head = 'EKLENEN PART (Excellence Bölümü - Hakemlerin "Veri toplamadaki gecikmeler temellendirilmemiş" eleştirisini çürütmek için QA/QC ve GAN eklenmiştir)'
    exc_body = """Autonomous QA/QC Pipelines and Algorithmic Mitigation of Data Acquisition Delays
To address the inherent risks of data acquisition delays and heterogeneous image quality from multi-center clinical field environments, the methodology is strictly decoupled from manual, sequential data grading. An automated Quality Assurance and Quality Control (QA/QC) pipeline will be executed pre-inference. For raw B-mode ultrasound and SWE data, a lightweight Convolutional Neural Network (MobileNetV3-Small) will evaluate transducer coupling and acoustic shadowing, outputting a structural similarity index measure (SSIM). Scans scoring an SSIM <0.75 are autonomously rejected, preventing the contamination of the downstream predictive pipeline.

Furthermore, any longitudinal data acquisition delays or incomplete multimodal matrices will not stall the machine learning workflow. To decouple model training from clinical delays, the architecture employs a sparsity-aware XGBoost framework, which natively handles missing tensors by learning default directional splits. In cases of severe multimodal data sparsity (e.g., missing MRI follow-ups), missing biomechanical and morphological vectors will be imputed using a pre-trained Generative Adversarial Network (GAN) architecture specifically optimized for tabular-radiomic imputation, ensuring the Elastic Net and XGBoost classifiers continuously receive dense, orthogonal matrices for ongoing hyperparameter tuning without waiting for full cohort completion."""

    imp_head = 'EKLENEN PART (Impact Bölümü - Hakemlerin "Fikri Mülkiyet (IP) koruması ve ticarileşme stratejisi yetersiz" eleştirisini çürütmek için SaMD mimarisi eklenmiştir)'
    imp_body = """Intellectual Property (IP) Protection and Software as a Medical Device (SaMD) Commercialization Architecture
The 'Hamstring Risk Card' transcends a theoretical framework by being architected as a deployable, cloud-native Software as a Medical Device (SaMD). To strictly protect the project's intellectual property and prevent reverse-engineering of the predictive models, the core algorithmic assets—specifically the U-Net spatial weights and the XGBoost decision tree ensembles—will be legally and architecturally maintained as Trade Secrets. These models will be hosted in isolated, hardware-encrypted cloud enclaves (e.g., AWS Nitro Enclaves).

End-users (elite clubs) will not have access to the raw source code or model weights. Instead, integration will occur exclusively via containerized, rate-limited RESTful API endpoints. This robust backend separation inherently protects the core IP while facilitating a highly scalable Business-to-Business (B2B) Software as a Service (SaaS) commercialization route. Provisional patents (Invention Disclosures) will be filed specifically covering the proprietary cross-modality data fusion vectors (combining SWE elastograms with MRI meshes) prior to any open-access publication. The front-end dashboard will be licensed to elite clubs via secure OAuth2 authentication protocols, ensuring GDPR-compliant data transmission and creating a direct revenue-generation pipeline post-fellowship."""

    impl_head = 'EKLENEN PART (Implementation Bölümü - Hakemlerin "Kilometre taşları jenerik, zamanlama kısa" eleştirisini çürütmek için Agile MLOps Milestones eklenmiştir)'
    impl_body = """Agile MLOps Integration and High-Resolution Predictive Milestones
The assertion that the integration timeframe is constrained is mitigated by replacing traditional waterfall software development with an Agile Machine Learning Operations (MLOps) CI/CD (Continuous Integration / Continuous Deployment) pipeline. Model training is not deferred until the completion of prospective data collection; rather, the baseline algorithms will be pre-trained on the retrospective SIRP-600 dataset and iteratively updated via dynamic retraining triggers.

To ensure granular tracking of the data science and computer vision pipelines, the following highly specific technical milestones are integrated into the 36-month timeline:
- M6 (V1 Model Freeze): Initial hyperparameter optimization (via GridSearchCV) and feature selection (Boruta algorithm) completed on the retrospective SIRP-600 dataset. Baseline XGBoost AUC established.
- M12 (Cross-Modality Pipeline Deployed): Autonomous spatial registration scripts mapping portable ultrasound coordinates to MRI macroscopic meshes validated (SSIM >0.85).
- M18 (Prospective Ingestion Checkpoint): First batch of prospective multimodal vectors ingested via the QA/QC API; concept drift metrics evaluated to trigger automatic model weight recalibration.
- M24 (External Validation Complete): Out-of-Fold (OOF) cross-validation on the secondary geographic cohort (Maynooth) finalised, verifying model robustness against demographic variance.
- M36 (SaMD API Hardening): Final XGBoost and U-Net RESTful endpoints stress-tested, load-balanced, and secured for commercial clinical deployment across elite clubs."""

    wp1_found = False
    wp3_found = False

    def add_to_paragraph_with_red(p, match_str, header_txt, body_txt):
        if match_str in p.text:
            add_red_header(p, header_txt)
            p.add_run(body_txt)
            return True
        return False

    for p in doc.paragraphs:
        text = p.text.strip().lower()
        if "1.2\tsoundness of the proposed methodology" in text:
            add_colored_injection_before(p, exc_head, exc_body)
        elif "2.2\tsuitability and quality of the measures to maximise expected outcomes" in text:
            add_colored_injection_before(p, imp_head, imp_body)
        elif "3.1\tquality and effectiveness of the work plan" in text:
            add_colored_injection_before(p, impl_head, impl_body)

        if not wp1_found and add_to_paragraph_with_red(p, "O1 - Establish a harmonised", wp1_head, wp1_body):
            wp1_found = True
        if not wp3_found and add_to_paragraph_with_red(p, "O3 - Develop and independently evaluate", wp3_head, wp3_body):
            wp3_found = True

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    if not wp1_found and add_to_paragraph_with_red(p, "O1 - Establish a harmonised", wp1_head, wp1_body):
                        wp1_found = True
                    if not wp3_found and add_to_paragraph_with_red(p, "O3 - Develop and independently evaluate", wp3_head, wp3_body):
                        wp3_found = True

    try:
        doc.save(output_file)
        print(f"File saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    process_file()
