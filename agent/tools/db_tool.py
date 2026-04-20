"""
DB query tool — in-memory breed knowledge base.

Provides structured, factual data about the 37 Oxford Pets breeds without
requiring an external service. This acts as a fast first-pass lookup before
falling back to web_search for topics not covered here.

In Phase 6, this module will be replaced (or supplemented) by a real
vector database (Pinecone / pgvector) that supports semantic retrieval.
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static knowledge base — 37 Oxford Pets breeds.
# Fields: description, origin, temperament, health_notes, lifespan, size.
# ---------------------------------------------------------------------------
_BREED_DB: dict[str, dict] = {
    # --- Cats ---
    "abyssinian": {
        "type": "cat",
        "description": "Slender, athletic cat with a ticked tabby coat.",
        "origin": "Ethiopia / UK",
        "temperament": "Active, curious, playful, affectionate",
        "health_notes": "Prone to progressive retinal atrophy and renal amyloidosis.",
        "lifespan": "9-15 years",
        "size": "small-medium",
    },
    "bengal": {
        "type": "cat",
        "description": "Wild-looking domestic cat with leopard-like spots or marbling.",
        "origin": "USA (domestic × Asian leopard cat)",
        "temperament": "Energetic, intelligent, vocal, loves water",
        "health_notes": "Risk of hypertrophic cardiomyopathy (HCM) and progressive retinal atrophy.",
        "lifespan": "12-16 years",
        "size": "medium-large",
    },
    "birman": {
        "type": "cat",
        "description": "Colour-pointed cat with silky semi-long coat and white gloves.",
        "origin": "Burma / France",
        "temperament": "Gentle, calm, social, good with children",
        "health_notes": "Generally healthy; watch for HCM.",
        "lifespan": "12-16 years",
        "size": "medium",
    },
    "bombay": {
        "type": "cat",
        "description": "Jet-black shorthair nicknamed the 'mini panther'.",
        "origin": "USA",
        "temperament": "Affectionate, people-oriented, playful",
        "health_notes": "Prone to craniofacial defect in kittens; watch for HCM.",
        "lifespan": "12-16 years",
        "size": "medium",
    },
    "british_shorthair": {
        "type": "cat",
        "description": "Stocky, round-faced cat with dense plush coat.",
        "origin": "UK",
        "temperament": "Calm, easygoing, loyal but not clingy",
        "health_notes": "Prone to HCM and polycystic kidney disease (PKD).",
        "lifespan": "12-17 years",
        "size": "large",
    },
    "egyptian_mau": {
        "type": "cat",
        "description": "Spotted shorthair — one of the few naturally spotted domestic breeds.",
        "origin": "Egypt",
        "temperament": "Shy with strangers, loyal to family, very fast",
        "health_notes": "Generally robust; watch for leukodystrophy.",
        "lifespan": "12-15 years",
        "size": "medium",
    },
    "maine_coon": {
        "type": "cat",
        "description": "Large, tufted-eared cat with shaggy semi-long coat.",
        "origin": "USA (Maine)",
        "temperament": "Dog-like, playful, gentle giant, good with kids and dogs",
        "health_notes": "Prone to HCM, spinal muscular atrophy, and hip dysplasia.",
        "lifespan": "12-15 years",
        "size": "large",
    },
    "persian": {
        "type": "cat",
        "description": "Flat-faced cat with very long, thick coat.",
        "origin": "Iran / UK",
        "temperament": "Quiet, gentle, affectionate, low-energy",
        "health_notes": "Brachycephalic airway issues, PKD, dental problems, eye discharge.",
        "lifespan": "12-17 years",
        "size": "medium-large",
    },
    "ragdoll": {
        "type": "cat",
        "description": "Large colour-pointed cat that goes limp when held.",
        "origin": "USA (California)",
        "temperament": "Docile, affectionate, follows owners room to room",
        "health_notes": "Risk of HCM and bladder stones.",
        "lifespan": "12-17 years",
        "size": "large",
    },
    "russian_blue": {
        "type": "cat",
        "description": "Elegant blue-grey cat with vivid green eyes.",
        "origin": "Russia",
        "temperament": "Reserved with strangers, devoted to family, quiet",
        "health_notes": "Generally very healthy; tendency to overeat — manage weight.",
        "lifespan": "15-20 years",
        "size": "medium",
    },
    "siamese": {
        "type": "cat",
        "description": "Sleek colour-pointed cat with striking blue eyes.",
        "origin": "Thailand",
        "temperament": "Vocal, social, demanding, highly intelligent",
        "health_notes": "Prone to amyloidosis, dental disease, and respiratory issues.",
        "lifespan": "12-20 years",
        "size": "medium",
    },
    "sphynx": {
        "type": "cat",
        "description": "Hairless (or nearly hairless) cat with wrinkled skin.",
        "origin": "Canada",
        "temperament": "Extremely affectionate, energetic, extroverted",
        "health_notes": "HCM, skin sunburn, earwax buildup, temperature sensitivity.",
        "lifespan": "8-14 years",
        "size": "medium",
    },
    # --- Dogs ---
    "american_bulldog": {
        "type": "dog",
        "description": "Muscular, powerful working dog descended from the Old English Bulldog.",
        "origin": "USA",
        "temperament": "Loyal, confident, energetic, assertive",
        "health_notes": "Hip/elbow dysplasia, cherry eye, brachycephalic syndrome.",
        "lifespan": "10-16 years",
        "size": "large",
    },
    "american_pit_bull_terrier": {
        "type": "dog",
        "description": "Medium-sized, muscular terrier — athletic and people-oriented.",
        "origin": "USA / UK",
        "temperament": "Loyal, affectionate with family, energetic, strong-willed",
        "health_notes": "Hip dysplasia, skin allergies, heart disease.",
        "lifespan": "12-16 years",
        "size": "medium",
    },
    "basset_hound": {
        "type": "dog",
        "description": "Low-set hound with long ears and exceptional sense of smell.",
        "origin": "France / Belgium",
        "temperament": "Gentle, patient, stubborn, good with children",
        "health_notes": "Ear infections, obesity, intervertebral disc disease, bloat.",
        "lifespan": "10-12 years",
        "size": "medium",
    },
    "beagle": {
        "type": "dog",
        "description": "Compact scent hound with floppy ears and a merry temperament.",
        "origin": "UK",
        "temperament": "Curious, friendly, merry, determined",
        "health_notes": "Epilepsy, hip dysplasia, hypothyroidism, ear infections.",
        "lifespan": "10-15 years",
        "size": "small-medium",
    },
    "boxer": {
        "type": "dog",
        "description": "Medium-large working dog with a square muzzle and playful energy.",
        "origin": "Germany",
        "temperament": "Playful, loyal, patient with children, protective",
        "health_notes": "Prone to cancer, heart issues (aortic stenosis, cardiomyopathy), hip dysplasia.",
        "lifespan": "10-12 years",
        "size": "large",
    },
    "chihuahua": {
        "type": "dog",
        "description": "Smallest recognised dog breed with a big personality.",
        "origin": "Mexico",
        "temperament": "Alert, bold, loyal, sometimes wary of strangers",
        "health_notes": "Patellar luxation, dental crowding, hypoglycemia, tracheal collapse.",
        "lifespan": "12-20 years",
        "size": "small",
    },
    "english_cocker_spaniel": {
        "type": "dog",
        "description": "Merry, active gundog with silky coat and long ears.",
        "origin": "UK",
        "temperament": "Affectionate, playful, eager to please",
        "health_notes": "Familial nephropathy, progressive retinal atrophy, ear infections.",
        "lifespan": "12-14 years",
        "size": "medium",
    },
    "english_setter": {
        "type": "dog",
        "description": "Elegant gundog with a speckled 'belton' coat.",
        "origin": "UK",
        "temperament": "Gentle, friendly, energetic, good family dog",
        "health_notes": "Hip dysplasia, hypothyroidism, deafness (in heavily mottled dogs).",
        "lifespan": "12 years",
        "size": "large",
    },
    "german_shorthaired": {
        "type": "dog",
        "description": "Versatile hunting dog, athletic and eager.",
        "origin": "Germany",
        "temperament": "Intelligent, energetic, friendly, trainable",
        "health_notes": "Hip dysplasia, bloat (GDV), lymphedema.",
        "lifespan": "12-14 years",
        "size": "large",
    },
    "great_pyrenees": {
        "type": "dog",
        "description": "Large, white livestock-guardian dog with thick double coat.",
        "origin": "France / Spain",
        "temperament": "Calm, patient, independent, protective",
        "health_notes": "Hip dysplasia, bloat, bone cancer.",
        "lifespan": "10-12 years",
        "size": "giant",
    },
    "havanese": {
        "type": "dog",
        "description": "Small, silky-coated companion dog — national dog of Cuba.",
        "origin": "Cuba",
        "temperament": "Playful, outgoing, affectionate, good for apartments",
        "health_notes": "Patellar luxation, hip dysplasia, progressive retinal atrophy, deafness.",
        "lifespan": "14-16 years",
        "size": "small",
    },
    "japanese_chin": {
        "type": "dog",
        "description": "Small, cat-like companion dog with a flat face.",
        "origin": "Japan (originally China)",
        "temperament": "Charming, loyal, quiet, agile",
        "health_notes": "Brachycephalic syndrome, heart murmurs, GM2 gangliosidosis.",
        "lifespan": "10-12 years",
        "size": "small",
    },
    "keeshond": {
        "type": "dog",
        "description": "Fluffy, medium-sized spitz-type dog with 'spectacles' eye markings.",
        "origin": "Netherlands",
        "temperament": "Friendly, lively, alert, good watchdog",
        "health_notes": "Hip dysplasia, epilepsy, Cushing's disease.",
        "lifespan": "12-15 years",
        "size": "medium",
    },
    "leonberger": {
        "type": "dog",
        "description": "Giant lion-like dog with a mane and gentle giant temperament.",
        "origin": "Germany",
        "temperament": "Gentle, confident, sociable, loves water",
        "health_notes": "Hip/elbow dysplasia, Leonberger polyneuropathy (LPN), cancer.",
        "lifespan": "8-9 years",
        "size": "giant",
    },
    "miniature_pinscher": {
        "type": "dog",
        "description": "Compact, fearless 'King of Toys' with high-stepping gait.",
        "origin": "Germany",
        "temperament": "Energetic, assertive, clever, escape artist",
        "health_notes": "Patellar luxation, Legg-Calvé-Perthes, progressive retinal atrophy.",
        "lifespan": "12-16 years",
        "size": "small",
    },
    "newfoundland": {
        "type": "dog",
        "description": "Massive, sweet-natured working dog — legendary water rescue breed.",
        "origin": "Canada",
        "temperament": "Sweet, patient, devoted, gentle with children",
        "health_notes": "Hip/elbow dysplasia, subvalvular aortic stenosis, bloat, cystinuria.",
        "lifespan": "8-10 years",
        "size": "giant",
    },
    "pomeranian": {
        "type": "dog",
        "description": "Tiny spitz-type dog with a profuse double coat and fox-like face.",
        "origin": "Germany / Poland",
        "temperament": "Lively, bold, curious, vocal",
        "health_notes": "Alopecia X, patellar luxation, tracheal collapse, dental issues.",
        "lifespan": "12-16 years",
        "size": "small",
    },
    "pug": {
        "type": "dog",
        "description": "Compact, wrinkly-faced companion dog with a curled tail.",
        "origin": "China",
        "temperament": "Charming, mischievous, loving, sociable",
        "health_notes": "Brachycephalic airway syndrome, eye ulcers, hip dysplasia, obesity.",
        "lifespan": "12-15 years",
        "size": "small",
    },
    "saint_bernard": {
        "type": "dog",
        "description": "Giant Alpine rescue dog, famously gentle and patient.",
        "origin": "Switzerland / Italy",
        "temperament": "Friendly, patient, gentle, good with children",
        "health_notes": "Hip/elbow dysplasia, bloat, osteosarcoma, entropion.",
        "lifespan": "8-10 years",
        "size": "giant",
    },
    "samoyed": {
        "type": "dog",
        "description": "White, fluffy spitz with a permanent 'Sammy smile'.",
        "origin": "Siberia",
        "temperament": "Friendly, gentle, adaptable, loves cold weather",
        "health_notes": "Hip dysplasia, Samoyed hereditary glomerulopathy, diabetes.",
        "lifespan": "12-14 years",
        "size": "medium-large",
    },
    "scottish_terrier": {
        "type": "dog",
        "description": "Compact, short-legged terrier with a distinctive beard and eyebrows.",
        "origin": "Scotland",
        "temperament": "Independent, feisty, loyal, dignified",
        "health_notes": "Scottie cramp, von Willebrand disease, bladder cancer, Cushing's.",
        "lifespan": "11-13 years",
        "size": "small",
    },
    "shiba_inu": {
        "type": "dog",
        "description": "Japan's most popular native breed — fox-like with a spirited personality.",
        "origin": "Japan",
        "temperament": "Alert, bold, independent, loyal",
        "health_notes": "Patellar luxation, hip dysplasia, allergies, glaucoma.",
        "lifespan": "13-16 years",
        "size": "medium",
    },
    "staffordshire_bull_terrier": {
        "type": "dog",
        "description": "Muscular, courageous terrier — nicknamed the 'nanny dog' for its gentleness with children.",
        "origin": "UK",
        "temperament": "Bold, trustworthy, affectionate, playful",
        "health_notes": "Hip dysplasia, hereditary cataracts (HC), L-2-HGA (neurological).",
        "lifespan": "12-14 years",
        "size": "medium",
    },
    "wheaten_terrier": {
        "type": "dog",
        "description": "Soft-coated Irish farm dog with a silky, wavy wheat-coloured coat.",
        "origin": "Ireland",
        "temperament": "Happy, spirited, devoted, energetic",
        "health_notes": "Protein-losing nephropathy (PLN), protein-losing enteropathy (PLE), Addison's.",
        "lifespan": "12-14 years",
        "size": "medium",
    },
    "yorkshire_terrier": {
        "type": "dog",
        "description": "Tiny terrier with a floor-length silky steel-blue and tan coat.",
        "origin": "UK (Yorkshire)",
        "temperament": "Feisty, brave, affectionate, vocal",
        "health_notes": "Patellar luxation, tracheal collapse, hypoglycemia, dental crowding.",
        "lifespan": "13-16 years",
        "size": "small",
    },
}

# Build a lowercase normalisation map: handle spaces vs underscores, capitalisation.
_NORMALISED: dict[str, str] = {
    k.lower().replace(" ", "_"): k for k in _BREED_DB
}


def _lookup(breed_name: str) -> Optional[dict]:
    """Case-insensitive, space/underscore-tolerant breed lookup."""
    key = breed_name.lower().strip().replace(" ", "_").replace("-", "_")
    canonical = _NORMALISED.get(key)
    if canonical:
        return _BREED_DB[canonical]
    # Partial match fallback: return first breed whose name contains the query.
    for norm_key, canonical in _NORMALISED.items():
        if key in norm_key or norm_key in key:
            return _BREED_DB[canonical]
    return None


@tool
def db_query(breed_name: str) -> str:
    """
    Query the local breed knowledge base for structured information about a
    cat or dog breed from the Oxford Pets dataset.

    Use this tool immediately after cv_predict identifies a breed, or when
    the user asks directly about one of the 37 Oxford Pets breeds.
    For topics not covered here (e.g. training tips, recent research),
    follow up with web_search.

    Args:
        breed_name: The breed name (e.g. "Siamese", "beagle", "Maine Coon").

    Returns:
        JSON string with breed details: type, description, origin, temperament,
        health_notes, lifespan, size.
        Returns an "error" field if the breed is not found in the database.
    """
    logger.info(f"[db_query] Looking up: {breed_name!r}")

    result = _lookup(breed_name)

    if result is None:
        available = sorted(_BREED_DB.keys())
        msg = (
            f"Breed '{breed_name}' not found in local knowledge base. "
            f"Available breeds: {', '.join(available)}. "
            "Try web_search for breeds outside the Oxford Pets dataset."
        )
        logger.warning(f"[db_query] {msg}")
        return json.dumps({"error": msg})

    logger.info(f"[db_query] Found: {result['type']} — {breed_name}")
    return json.dumps({"breed": breed_name, **result})
