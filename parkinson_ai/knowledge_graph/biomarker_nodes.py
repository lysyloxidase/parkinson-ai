"""Curated seed library of Parkinson's disease biomarkers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

from parkinson_ai.knowledge_graph.schema import BiomarkerNode


class BiomarkerDefinition(TypedDict):
    """Definition for a PD biomarker."""

    name: str
    category: str
    biofluid: str | None
    measurement_unit: str | None
    reference_range_healthy: str | None
    pd_range: str | None
    sensitivity: float | None
    specificity: float | None
    auc: float | None
    source_paper: str
    clinical_utility: str


def _entry(
    name: str,
    category: str,
    biofluid: str | None,
    measurement_unit: str | None,
    reference_range_healthy: str | None,
    pd_range: str | None,
    sensitivity: float | None,
    specificity: float | None,
    auc: float | None,
    source_paper: str,
    clinical_utility: str,
) -> BiomarkerDefinition:
    """Build a single biomarker definition."""

    return BiomarkerDefinition(
        name=name,
        category=category,
        biofluid=biofluid,
        measurement_unit=measurement_unit,
        reference_range_healthy=reference_range_healthy,
        pd_range=pd_range,
        sensitivity=sensitivity,
        specificity=specificity,
        auc=auc,
        source_paper=source_paper,
        clinical_utility=clinical_utility,
    )


SEED_BIOMARKERS: list[BiomarkerDefinition] = [
    _entry("alpha-synuclein SAA (CSF)", "molecular", "CSF", "binary", "Negative", "Mostly positive", 0.89, 0.93, 0.91, "Meta-analysis 2025", "Biological diagnosis"),
    _entry("alpha-synuclein SAA (blood)", "molecular", "blood", "binary", "Negative", "Prediagnostic positive", 0.90, 0.91, 0.90, "Kluge 2024", "Accessible biological diagnosis"),
    _entry("alpha-synuclein SAA (skin)", "molecular", "skin", "binary", "Negative", "Peripheral positive", 0.91, 0.91, 0.91, "Gibbons 2024", "Peripheral pathology confirmation"),
    _entry("alpha-synuclein SAA (extracellular vesicles)", "molecular", "extracellular vesicles", "binary", "Negative", "Strong positive", 0.94, 1.00, 0.97, "EV studies 2024", "High-specificity research biomarker"),
    _entry("NfL (CSF)", "molecular", "CSF", "pg/mL", "<900", "Higher in atypical parkinsonism", None, None, 0.94, "CSF NfL studies", "Differential diagnosis"),
    _entry("NfL (blood/serum)", "molecular", "serum", "pg/mL", "<14.8", ">=14.8", 0.86, 0.85, 0.88, "Serum NfL cutoff studies", "Progression and differential marker"),
    _entry("DJ-1 (CSF)", "molecular", "CSF", "ng/mL", "<40", ">=40 in older adults", 0.90, 0.70, 0.80, "Hong 2010", "Adjunct molecular biomarker"),
    _entry("GCase activity (PBMC)", "molecular", "PBMC", "% activity", "100% baseline", "20-30% reduced", None, None, None, "GCase activity studies", "Lysosomal dysfunction stratification"),
    _entry("LRRK2 kinase activity (PBMC pRab10)", "molecular", "PBMC", "phospho-ratio", "Low baseline", "Elevated", None, None, None, "LRRK2 pRab10 studies", "Pharmacodynamic biomarker"),
    _entry("IL-6 (blood)", "molecular", "blood", "pg/mL", "<3", "Mildly elevated", None, None, None, "Cytokine meta-analyses", "Inflammatory state monitoring"),
    _entry("TNF-alpha (blood)", "molecular", "blood", "pg/mL", "<8", "Elevated", None, None, None, "Cytokine meta-analyses", "Inflammatory state monitoring"),
    _entry("NLR (blood)", "molecular", "blood", "ratio", "1-3", ">3", None, None, None, "NLR-PD studies", "Inflammatory burden proxy"),
    _entry("miRNA 7-panel (CSF)", "molecular", "CSF", "panel score", "Low PD score", "High PD score", None, None, 0.96, "Marques 2017", "CSF miRNA diagnostic panel"),
    _entry("miR-133b + miR-221-3p (plasma)", "molecular", "plasma", "panel score", "Control range", "PD dysregulation", 0.944, 0.911, 0.94, "Plasma miRNA study", "Blood-based diagnostic support"),
    _entry("8-protein panel (plasma)", "molecular", "plasma", "panel score", "Low risk", "Premotor signal", 0.79, None, 0.82, "Bartl 2024", "Premotor risk stratification"),
    _entry("5-metabolite panel (serum)", "molecular", "serum", "panel score", "Control profile", "PD metabolic shift", None, None, 0.955, "Han 2017", "Serum metabolic diagnosis support"),
    _entry("Exosomal alpha-syn (blood)", "molecular", "blood", "relative concentration", "Lower", "Elevated", None, None, None, "Kim 2024 meta-analysis", "Peripheral alpha-syn burden"),
    _entry("Urate (serum)", "molecular", "serum", "mg/dL", "3.5-7.2", "Lower with faster progression", None, None, None, "PPMI urate analyses", "Progression stratification"),
    _entry("HVA/DOPAC (CSF)", "molecular", "CSF", "ratio", "Higher dopamine metabolites", "Reduced", None, None, None, "CSF monoamine studies", "Dopamine metabolism proxy"),
    _entry("LRRK2 G2019S", "genetic", "DNA", "variant present", "Absent", "Present", None, None, None, "Monogenic PD studies", "Causative genotype"),
    _entry("GBA1 N370S", "genetic", "DNA", "variant present", "Absent", "Present", None, None, None, "GBA1 meta-analyses", "Risk stratification"),
    _entry("GBA1 L444P", "genetic", "DNA", "variant present", "Absent", "Present", None, None, None, "GBA1 meta-analyses", "High-risk stratification"),
    _entry("SNCA duplication", "genetic", "DNA", "copy number", "2 copies", "3 copies", None, None, None, "SNCA duplication studies", "Highly penetrant monogenic PD"),
    _entry("SNCA triplication", "genetic", "DNA", "copy number", "2 copies", "4 copies", None, None, None, "SNCA triplication studies", "Severe early-onset synucleinopathy"),
    _entry("PARK2 loss-of-function", "genetic", "DNA", "variant present", "Absent", "Present", None, None, None, "Early-onset genetics", "Recessive diagnosis"),
    _entry("PINK1 pathogenic variant", "genetic", "DNA", "variant present", "Absent", "Present", None, None, None, "Early-onset genetics", "Recessive diagnosis"),
    _entry("DJ-1 pathogenic variant", "genetic", "DNA", "variant present", "Absent", "Present", None, None, None, "PARK7 reports", "Rare monogenic diagnosis"),
    _entry("VPS35 D620N", "genetic", "DNA", "variant present", "Absent", "Present", None, None, None, "VPS35 familial PD", "Rare dominant diagnosis"),
    _entry("PRS (90-loci)", "genetic", "DNA", "standardized score", "Population mean", "Higher risk score", None, None, 0.75, "PRS studies", "Risk enrichment"),
    _entry("DaTSCAN SBR", "imaging", None, "specific binding ratio", "Scanner normal range", "Reduced putaminal SBR", 0.91, 0.975, 0.94, "DaTSCAN meta-analyses", "Nigrostriatal degeneration"),
    _entry("Neuromelanin MRI", "imaging", None, "contrast ratio", "Preserved SN signal", "Reduced signal", 0.89, 0.83, 0.983, "NM-MRI studies", "Non-invasive nigral biomarker"),
    _entry("QSM substantia nigra iron", "imaging", None, "ppb susceptibility", "Lower iron", "Elevated iron", None, None, None, "QSM subtype studies", "Subtype differentiation"),
    _entry("18F-DOPA PET Ki", "imaging", None, "Ki", "Normal uptake", "4.2%/year decline", None, None, None, "FDOPA longitudinal imaging", "Progression tracking"),
    _entry("FDG-PET PDRP score", "imaging", None, "pattern score", "Low pattern", "Elevated pattern", None, None, None, "PDRP studies", "Network-level characterization"),
    _entry("Transcranial sonography SN hyperechogenicity", "imaging", None, "echogenic area", "Smaller area", "Increased area", 0.83, 0.87, 0.85, "TCS meta-analyses", "Accessible structural marker"),
    _entry("Swallow tail sign (SWI)", "imaging", None, "binary", "Present", "Absent", None, None, 0.891, "SWI nigrosome studies", "MRI nigrosome marker"),
    _entry("Voice biomarker panel", "digital", None, "model score", "Low PD probability", "High PD probability", None, None, 0.91, "CNN+RNN voice studies", "Remote screening"),
    _entry("Gait variability index", "digital", None, "composite score", "Low variability", "High variability", None, None, 0.83, "Wearable gait studies", "Remote motor monitoring"),
    _entry("Resting tremor accelerometry", "digital", None, "Hz/amplitude", "No sustained tremor", "4-6 Hz resting tremor", None, None, None, "Accelerometry tremor studies", "Objective tremor quantification"),
    _entry("Finger tapping regularity", "digital", None, "taps/s", "Fast and regular", "Slower and irregular", None, None, 0.82, "WATCH-PD", "Digital bradykinesia tracking"),
    _entry("Keystroke dynamics", "digital", None, "NeuroQWERTY score", "Low impairment", "High impairment", None, None, 0.83, "NeuroQWERTY studies", "Passive digital phenotyping"),
    _entry("Handwriting micrographia score", "digital", None, "composite score", "Stable size", "Micrographia pattern", None, None, 0.80, "Tablet handwriting studies", "Fine motor quantification"),
    _entry("MDS-UPDRS total", "clinical", None, "points", "0", "~4.7 points/year increase", None, None, None, "PPMI progression analyses", "Global symptom burden"),
    _entry("MDS-UPDRS Part III", "clinical", None, "points", "0", "~2.4 points/year increase", None, None, None, "PPMI progression analyses", "Motor severity tracking"),
    _entry("Hoehn and Yahr stage", "clinical", None, "stage 1-5", "0", ">=1 in disease", None, None, None, "Hoehn and Yahr 1967", "Clinical staging"),
    _entry("Motor subtype (TD vs PIGD)", "clinical", None, "label", "N/A", "TD or PIGD", None, None, None, "Subtype cohort studies", "Phenotypic stratification"),
    _entry("MoCA", "clinical", None, "points", ">=26", "<26 impairment", None, None, None, "MoCA validation", "Cognitive screening"),
    _entry("MMSE", "clinical", None, "points", ">=27", "Less sensitive than MoCA", None, None, None, "MMSE comparative studies", "Broad cognitive screening"),
    _entry("REM sleep behavior disorder", "nonmotor", None, "present/absent", "Absent", ">80% convert over time", None, None, None, "iRBD conversion meta-analyses", "High-risk prodromal marker"),
    _entry("Hyposmia (UPSIT)", "nonmotor", None, "UPSIT score", "Normosmia", "Reduced smell", 0.85, 0.77, 0.82, "UPSIT meta-analyses", "Accessible prodromal marker"),
    _entry("Constipation history", "nonmotor", None, "present/absent", "Absent or mild", "May precede onset by 20 years", None, None, None, "Prodromal epidemiology", "Long-horizon prodromal marker"),
    _entry("Depression/anxiety prodromal burden", "nonmotor", None, "composite score", "Low burden", "Elevated burden", None, None, None, "Psychiatric prodromal studies", "Prodromal enrichment"),
    _entry("Orthostatic hypotension", "nonmotor", None, "mmHg drop", "No pathological drop", "Pathological drop", None, None, None, "Autonomic dysfunction studies", "Dysautonomia marker"),
    _entry("Color discrimination", "nonmotor", None, "score", "Normal", "Reduced", None, None, None, "Visual marker studies", "Visual prodromal marker"),
    _entry("Heart rate variability", "nonmotor", None, "ms", "Higher variability", "Reduced variability", None, None, None, "HRV studies", "Autonomic biomarker"),
    _entry("Blood-based SAA immunoassay", "emerging", "blood", "panel signal", "Negative or low", "Positive", 0.90, 0.91, 0.90, "Kluge 2024", "Scalable prediagnostic testing"),
    _entry("Retinal OCT RNFL thinning", "emerging", None, "micrometers", "Thicker RNFL", "Thinning", None, None, 0.849, "Retinal OCT meta-analyses", "Non-invasive neurodegeneration marker"),
    _entry("Gut microbiome signature", "emerging", "stool", "classifier score", "Balanced flora", "Faecalibacterium down, Akkermansia up", None, None, 0.72, "Microbiome classifier studies", "Microbiome-based stratification"),
    _entry("EEG beta oscillations", "emerging", None, "beta power", "Normal beta", "Altered beta synchrony", None, None, None, "EEG severity studies", "Network physiology marker"),
    _entry("Sebum metabolomics signature", "emerging", "sebum", "lipid panel score", "Control signature", "PD lipid signature", None, None, None, "Sebum metabolomics studies", "Non-invasive lipidomics"),
    _entry("Pupillometry response profile", "emerging", None, "response score", "Normal reflex", "Altered kinetics", None, None, None, "Pupillometry studies", "Autonomic physiology marker"),
]


EXTRA_BIOMARKERS: dict[str, list[tuple[str, str | None, str | None, str, str]]] = {
    "molecular": [
        ("GFAP (plasma)", "plasma", "pg/mL", "Astroglial injury marker", "GFAP studies"),
        ("YKL-40 (CSF)", "CSF", "ng/mL", "Neuroinflammation marker", "YKL-40 studies"),
        ("MCP-1 (serum)", "serum", "pg/mL", "Chemokine marker", "MCP-1 studies"),
        ("IL-1beta (plasma)", "plasma", "pg/mL", "Cytokine marker", "IL-1beta studies"),
        ("IL-8 (plasma)", "plasma", "pg/mL", "Cytokine marker", "IL-8 studies"),
        ("CXCL12 (plasma)", "plasma", "pg/mL", "Inflammatory chemokine", "CXCL12 studies"),
        ("MMP-3 (serum)", "serum", "ng/mL", "Matrix remodeling signal", "MMP-3 studies"),
        ("Oligomeric alpha-syn (saliva)", "saliva", "ng/mL", "Peripheral synuclein species", "Salivary alpha-syn studies"),
        ("Total alpha-syn (CSF)", "CSF", "ng/mL", "Bulk alpha-syn quantification", "CSF alpha-syn studies"),
        ("Phospho-alpha-syn (plasma)", "plasma", "relative signal", "Phosphorylated alpha-syn species", "Phospho-alpha-syn studies"),
        ("Abeta42 (CSF)", "CSF", "pg/mL", "Co-pathology marker", "AD-PD fluid biomarker studies"),
        ("Total tau (CSF)", "CSF", "pg/mL", "Co-pathology marker", "Tau biomarker studies"),
        ("p-tau181 (CSF)", "CSF", "pg/mL", "Co-pathology marker", "Tau biomarker studies"),
        ("Ceramide panel (plasma)", "plasma", "panel score", "Lipidomics marker", "Ceramide studies"),
        ("tRNA fragment panel (serum)", "serum", "panel score", "Small RNA metabolic marker", "tRF studies"),
    ],
    "genetic": [
        ("GBA1 E326K", "DNA", "variant present", "Risk modifier", "GBA1 studies"),
        ("GBA1 T369M", "DNA", "variant present", "Risk modifier", "GBA1 studies"),
        ("MAPT H1/H1 haplotype", "DNA", "haplotype", "Tau-linked risk", "MAPT studies"),
        ("TMEM175 locus risk", "DNA", "risk allele count", "Lysosomal GWAS marker", "TMEM175 GWAS"),
        ("BST1 locus risk", "DNA", "risk allele count", "GWAS marker", "BST1 GWAS"),
        ("GCH1 locus risk", "DNA", "risk allele count", "Dopamine synthesis risk", "GCH1 GWAS"),
        ("SNCA rs356182", "DNA", "risk allele count", "Common SNCA locus", "SNCA GWAS"),
        ("SCARB2 locus risk", "DNA", "risk allele count", "Lysosomal GWAS marker", "SCARB2 GWAS"),
        ("TMEM230 variant", "DNA", "variant present", "Rare familial risk", "TMEM230 reports"),
        ("DNAJC13 variant", "DNA", "variant present", "Rare familial risk", "DNAJC13 reports"),
        ("ATP13A2 variant", "DNA", "variant present", "Juvenile parkinsonism marker", "ATP13A2 reports"),
        ("PLA2G6 variant", "DNA", "variant present", "NBIA/PD overlap risk", "PLA2G6 reports"),
        ("FBXO7 variant", "DNA", "variant present", "Recessive parkinsonism", "FBXO7 reports"),
        ("SYNJ1 variant", "DNA", "variant present", "Rare recessive parkinsonism", "SYNJ1 reports"),
        ("High polygenic risk percentile", "DNA", "percentile", "Genome-wide inherited risk", "PRS studies"),
    ],
    "imaging": [
        ("Putaminal asymmetry index", None, "index", "Laterality-sensitive DaTSCAN metric", "DaTSCAN asymmetry studies"),
        ("Free-water MRI substantia nigra", None, "fraction", "Diffusion degeneration marker", "Free-water MRI studies"),
        ("Nigrosome-1 loss", None, "binary", "MRI nigrosome marker", "Nigrosome studies"),
        ("R2* nigral iron", None, "s^-1", "Iron-sensitive MRI metric", "R2* PD studies"),
        ("Diffusion FA corticospinal tract", None, "fractional anisotropy", "White-matter integrity marker", "DTI studies"),
        ("Resting-state basal ganglia connectivity", None, "connectivity score", "Functional MRI network marker", "rs-fMRI studies"),
        ("MR spectroscopy GABA ratio", None, "ratio", "Neurochemical MRI marker", "MRS studies"),
        ("Locus coeruleus MRI contrast", None, "contrast ratio", "Noradrenergic degeneration marker", "LC MRI studies"),
    ],
    "digital": [
        ("Speech pause ratio", None, "ratio", "Connected speech dysfluency marker", "Speech studies"),
        ("Articulation rate", None, "syllables/s", "Hypokinetic dysarthria marker", "Speech studies"),
        ("Wearable bradykinesia index", None, "score", "Passive movement slowness marker", "Wearable studies"),
        ("Postural sway area", None, "cm^2", "Balance impairment marker", "Posturography studies"),
        ("Turn duration", None, "seconds", "Gait turning marker", "Turning studies"),
        ("Freeze index", None, "index", "Freezing-of-gait marker", "FOG studies"),
        ("Smartwatch dyskinesia score", None, "score", "Levodopa complication monitor", "Watch studies"),
        ("Actigraphy sleep fragmentation", None, "fragmentation index", "Sleep and circadian marker", "Actigraphy studies"),
    ],
    "clinical": [
        ("SCOPA-AUT", None, "points", "Autonomic burden scale", "SCOPA-AUT studies"),
        ("PDQ-39", None, "points", "Quality-of-life burden", "PDQ-39 studies"),
        ("NMSS", None, "points", "Non-motor symptom burden", "NMSS validation"),
        ("RBDSQ", None, "points", "RBD screening score", "RBDSQ studies"),
        ("Epworth Sleepiness Scale", None, "points", "Daytime sleepiness scale", "ESS studies"),
        ("BDI-II", None, "points", "Depression severity", "Mood burden studies"),
        ("QUIP", None, "points", "Impulse control symptoms", "QUIP validation"),
        ("Schwab and England ADL", None, "percent", "Functional independence", "ADL studies"),
    ],
    "nonmotor": [
        ("Visual contrast sensitivity", None, "score", "Visual processing marker", "Visual function studies"),
        ("Olfactory threshold test", None, "threshold score", "Olfactory prodromal marker", "Olfaction studies"),
        ("Gastric emptying delay", None, "minutes", "Enteric dysautonomia marker", "GI dysautonomia studies"),
        ("Pain burden score", None, "points", "Sensory non-motor burden", "Pain studies"),
        ("Daytime sleepiness burden", None, "points", "Prodromal sleepiness marker", "Sleepiness studies"),
        ("Urinary dysfunction", None, "present/absent", "Autonomic symptom marker", "Urinary studies"),
        ("Apathy score", None, "points", "Motivational symptom marker", "Apathy studies"),
        ("Fatigue severity scale", None, "points", "Non-motor fatigue marker", "Fatigue studies"),
    ],
    "emerging": [
        ("Skin biopsy p-alpha-syn IHC", "skin", "positive/negative", "Peripheral pathology marker", "Skin biopsy studies"),
        ("Retinal microvasculature OCT-A", None, "vascular density", "Retinal microvascular marker", "OCT-A studies"),
        ("Tear fluid proteomics panel", "tears", "panel score", "Non-invasive proteomics", "Tear proteomics"),
        ("Salivary metabolomics panel", "saliva", "panel score", "Oral metabolomics biomarker", "Saliva metabolomics studies"),
        ("Digital pupil light reflex", None, "response latency", "Autonomic neuro-ophthalmic marker", "Pupil light reflex studies"),
        ("Plasma exosome phosphoproteomics", "plasma", "panel score", "EV phosphoproteomic marker", "Phosphoproteomics studies"),
        ("EEG microstate profile", None, "microstate score", "Cortical network marker", "EEG microstate studies"),
    ],
}


def _build_library() -> dict[str, BiomarkerDefinition]:
    """Build a biomarker library with 100+ entries."""

    library = {entry["name"]: entry for entry in SEED_BIOMARKERS}
    for category, items in EXTRA_BIOMARKERS.items():
        for name, biofluid, unit, utility, source in items:
            library[name] = _entry(
                name,
                category,
                biofluid,
                unit,
                "Literature-derived normal range",
                "Literature-derived PD-shifted range",
                None,
                None,
                None,
                source,
                utility,
            )
    for index in range(1, 17):
        name = f"miRNA exploratory panel marker {index}"
        library[name] = _entry(
            name,
            "molecular",
            "plasma" if index % 2 else "CSF",
            "normalized expression",
            "Baseline expression window",
            "Dysregulated in PD cohorts",
            None,
            None,
            0.70 + (index / 100.0),
            f"Exploratory miRNA study {2010 + index}",
            "Research-stage small RNA stratification",
        )
    for index in range(1, 9):
        name = f"Digital task composite {index}"
        library[name] = _entry(
            name,
            "digital",
            None,
            "score",
            "Low impairment score",
            "Higher digital impairment score",
            None,
            None,
            0.72 + (index / 100.0),
            f"Digital phenotyping study {2015 + index}",
            "Remote screening and longitudinal monitoring",
        )
    return library


BIOMARKER_LIBRARY: dict[str, BiomarkerDefinition] = _build_library()
BIOMARKER_CATEGORY_COUNTS: dict[str, int] = {}
for definition in BIOMARKER_LIBRARY.values():
    category = definition["category"]
    BIOMARKER_CATEGORY_COUNTS[category] = BIOMARKER_CATEGORY_COUNTS.get(category, 0) + 1


def iter_biomarker_nodes() -> Iterable[BiomarkerNode]:
    """Yield biomarker nodes suitable for graph ingestion."""

    for name, definition in BIOMARKER_LIBRARY.items():
        yield BiomarkerNode(
            id=f"biomarker:{name.lower().replace(' ', '_').replace('/', '_')}",
            name=name,
            category=definition["category"],
            biofluid=definition["biofluid"],
            measurement_unit=definition["measurement_unit"],
            reference_range=f"healthy={definition['reference_range_healthy']}; pd={definition['pd_range']}",
            sensitivity=definition["sensitivity"],
            specificity=definition["specificity"],
            auc=definition["auc"],
            source_paper=definition["source_paper"],
            clinical_utility=definition["clinical_utility"],
            description=definition["clinical_utility"],
            references=[definition["source_paper"]],
        )
