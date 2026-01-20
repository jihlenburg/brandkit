#!/usr/bin/env python3
"""
Markov Chain Brand Name Generator
==================================
Generates brand names using character-level Markov chains trained on
a curated corpus of successful brand names.

Key features:
- Variable order (bigram, trigram, or interpolated)
- Temperature control for creativity vs. consistency
- Semantic seeding (start with meaningful morphemes)
- Phoneme-aware generation option
- Integration with scoring system

Theory:
-------
Markov chains model P(next_char | previous_n_chars). The "order" n determines
how much context is used:
- Order 1 (bigram): Too chaotic, many unrealistic combinations
- Order 2 (trigram): Good balance, learns common patterns
- Order 3 (4-gram): More conservative, closer to training data
- Interpolation: Weighted mix of orders for best of both worlds
"""

import random
import math
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =============================================================================
# TRAINING CORPUS (Expanded)
# =============================================================================

# Curated corpus of brand names organized by category and quality
# Total: ~600+ unique names across categories
TRAINING_CORPUS = {
    # Camping, RV, Caravan, and Motorhome brands (Europe + USA)
    # Sources: Erwin Hymer Group, Knaus Tabbert Group, independent manufacturers
    'camping': [
        # Erwin Hymer Group brands
        'Hymer', 'Dethleffs', 'Burstner', 'Carado', 'Sunlight',
        'Etrusco', 'Elddis', 'Laika', 'Niesmann', 'Eriba',
        # Knaus Tabbert Group
        'Knaus', 'Tabbert', 'Weinsberg', 'Morelo', 'Boxstar',
        # Other European manufacturers
        'Carthago', 'Fendt', 'Hobby', 'Frankia', 'Rapido',
        'Chausson', 'Pilote', 'Challenger', 'Autostar', 'Trigano',
        'Caravelair', 'Sterckeman', 'Kabe', 'Solifer', 'Adria',
        'Rimor', 'Mobilvetta', 'Roller', 'Benimar', 'Elnagh',
        'Malibu', 'Poessl', 'Globecar', 'Karmann', 'Westfalia',
        'Reimo', 'Vantourer', 'Bravia', 'Itineo', 'Florium',
        'Notin', 'Fleurette', 'Bavaria', 'Forster', 'Eura',
        # UK brands
        'Swift', 'Bailey', 'Coachman', 'Lunar', 'Compass',
        'Buccaneer', 'Sprite', 'Unicorn', 'Pageant', 'Eccles',
        # US brands
        'Airstream', 'Winnebago', 'Jayco', 'Keystone', 'Thor',
        'Fleetwood', 'Coachmen', 'Newmar', 'Tiffin', 'Entegra',
        'Monaco', 'Holiday', 'Prevost', 'Pleasure', 'Roadtrek',
        # Camping equipment
        'Dometic', 'Truma', 'Fiamma', 'Thule', 'Webasto',
        'Thetford', 'Waeco', 'Alde', 'Propex', 'Eberspacher',
        'Coleman', 'Campingaz', 'Primus', 'Jetboil', 'Optimus',
    ],

    # Energy, Solar, Power Electronics, DC-DC converters
    'energy': [
        # Portable power stations
        'Victron', 'Renogy', 'Ecoflow', 'Bluetti', 'Jackery',
        'Anker', 'Zendure', 'Poweroak', 'Fossibot', 'Allpowers',
        'Baldr', 'Pecron', 'Oukitel', 'Vtoman', 'Dabbsson',
        'Ugreen', 'Baseus', 'Rocksolar', 'Litime', 'Redodo',
        'Ampere', 'Lithium', 'Battleborn', 'Relion', 'Expion',
        # Solar panel manufacturers
        'Sunpower', 'Jinko', 'Trina', 'Longi', 'Risen',
        'Seraphim', 'Astronergy', 'Canadian', 'Hanwha', 'Jasolar',
        'Qcells', 'Axitec', 'Solarwatt', 'Meyer', 'Aleo',
        # Inverter/charger brands
        'Fronius', 'Kostal', 'Growatt', 'Goodwe', 'Huawei',
        'Sungrow', 'Sofar', 'Deye', 'Afore', 'Solis',
        'Enphase', 'Solaredge', 'Hoymiles', 'Apsystems', 'Tigo',
        # Battery manufacturers
        'Pylontech', 'Byd', 'Catl', 'Tesla', 'Sonnen',
        'Senec', 'Varta', 'Hoppecke', 'Banner', 'Exide',
        'Optima', 'Odyssey', 'Northstar', 'Deka', 'Trojan',
        # DC-DC specific (Victron Orion etc.)
        'Orion', 'Sterling', 'Redarc', 'Ctek', 'Noco',
        'Votronic', 'Schaudt', 'Calira', 'Nordelettronica', 'Cmtpower',
        # Legacy power brands
        'Duracell', 'Energizer', 'Panasonic', 'Samsung', 'Sanyo',
        'Maxell', 'Rayovac', 'Eveready', 'Procell', 'Varta',
    ],

    # Technology and Electronics brands (phonetically excellent)
    'tech': [
        # Consumer electronics
        'Nokia', 'Sony', 'Philips', 'Canon', 'Nikon',
        'Epson', 'Asus', 'Acer', 'Lenovo', 'Toshiba',
        'Fujitsu', 'Hitachi', 'Sharp', 'Casio', 'Seiko',
        'Citizen', 'Olympus', 'Pentax', 'Ricoh', 'Kyocera',
        'Pioneer', 'Kenwood', 'Denon', 'Marantz', 'Onkyo',
        'Yamaha', 'Roland', 'Korg', 'Bose', 'Sonos',
        'Harman', 'Jbl', 'Akg', 'Shure', 'Audio',
        # Computing
        'Dell', 'Hewlett', 'Nvidia', 'Intel', 'Qualcomm',
        'Broadcom', 'Marvell', 'Xilinx', 'Altera', 'Lattice',
        'Cypress', 'Maxim', 'Linear', 'Analog', 'Microchip',
        'Atmel', 'Infineon', 'Rohm', 'Murata', 'Vishay',
        'Omron', 'Keyence', 'Fanuc', 'Kuka', 'Yaskawa',
        # Wearables & GPS
        'Garmin', 'Suunto', 'Polar', 'Fitbit', 'Wahoo',
        'Coros', 'Amazfit', 'Xiaomi', 'Huawei', 'Oppo',
        'Vivo', 'Realme', 'Oneplus', 'Honor', 'Meizu',
    ],

    # Classic successful brands (phonetically pleasing, memorable)
    'classic': [
        # Sports & Fashion
        'Nike', 'Adidas', 'Puma', 'Reebok', 'Asics',
        'Fila', 'Kappa', 'Umbro', 'Lotto', 'Diadora',
        'Vans', 'Converse', 'Skechers', 'Brooks', 'Saucony',
        'Mizuno', 'Salomon', 'Merrell', 'Keen', 'Teva',
        # Lifestyle
        'Lego', 'Ikea', 'Zara', 'Mango', 'Uniqlo',
        'Muji', 'Miniso', 'Daiso', 'Nitori', 'Franc',
        # Tech giants
        'Amazon', 'Google', 'Apple', 'Meta', 'Netflix',
        'Spotify', 'Adobe', 'Oracle', 'Cisco', 'Vmware',
        'Salesforce', 'Workday', 'Servicenow', 'Atlassian', 'Zendesk',
        # Startups with great names
        'Uber', 'Lyft', 'Airbnb', 'Stripe', 'Shopify',
        'Slack', 'Zoom', 'Asana', 'Trello', 'Notion',
        'Figma', 'Canva', 'Miro', 'Linear', 'Vercel',
        'Netlify', 'Heroku', 'Render', 'Supabase', 'Planetscale',
        'Postman', 'Insomnia', 'Raycast', 'Linear', 'Loom',
    ],

    # Latin/Greek inspired names (timeless, international appeal)
    'classical': [
        # Celestial
        'Aurora', 'Aura', 'Stella', 'Luna', 'Nova',
        'Vega', 'Lyra', 'Andromeda', 'Cassiopeia', 'Polaris',
        'Sirius', 'Rigel', 'Altair', 'Deneb', 'Antares',
        'Capella', 'Arcturus', 'Aldebaran', 'Betelgeuse', 'Canopus',
        # Planets & Moons
        'Orion', 'Atlas', 'Titan', 'Europa', 'Io',
        'Callisto', 'Ganymede', 'Triton', 'Proteus', 'Nereid',
        'Mimas', 'Enceladus', 'Dione', 'Rhea', 'Tethys',
        # Mythology
        'Phoenix', 'Helios', 'Selene', 'Artemis', 'Apollo',
        'Hermes', 'Athena', 'Minerva', 'Jupiter', 'Saturn',
        'Neptune', 'Pluto', 'Venus', 'Mars', 'Mercury',
        'Ceres', 'Pallas', 'Juno', 'Vesta', 'Fortuna',
        'Victoria', 'Concordia', 'Pax', 'Libertas', 'Veritas',
        # Elements & Nature (Latin)
        'Terra', 'Aqua', 'Ignis', 'Ventus', 'Lumen',
        'Aether', 'Nox', 'Lux', 'Sol', 'Umbra',
        'Flora', 'Fauna', 'Silva', 'Arbor', 'Folia',
        'Petra', 'Ferrum', 'Aurum', 'Argentum', 'Cuprum',
        # Abstract concepts (Greek/Latin)
        'Chronos', 'Kairos', 'Aion', 'Cosmos', 'Logos',
        'Ethos', 'Pathos', 'Telos', 'Praxis', 'Gnosis',
        'Kinesis', 'Genesis', 'Metamorphosis', 'Synthesis', 'Analysis',
    ],

    # Outdoor & Adventure brands
    'outdoor': [
        # Premium outdoor
        'Patagonia', 'Arcteryx', 'Mammut', 'Fjallraven', 'Norrona',
        'Haglofs', 'Peak', 'Rab', 'Montane', 'Mountain',
        # Mass market outdoor
        'Northface', 'Columbia', 'Marmot', 'Osprey', 'Deuter',
        'Gregory', 'Kelty', 'Lowe', 'Vaude', 'Jack',
        # Technical gear
        'Black', 'Petzl', 'Grivel', 'Edelrid', 'Beal',
        'Metolius', 'Wild', 'Sterling', 'Tendon', 'Fixe',
        # Camping specific
        'Hilleberg', 'Exped', 'Thermarest', 'Nemo', 'Big',
        'Sierra', 'Msr', 'Snow', 'Jetboil', 'Soto',
        'Primus', 'Trangia', 'Esbit', 'Optimus', 'Kovea',
        # Footwear
        'Scarpa', 'Lowa', 'Meindl', 'Hanwag', 'Zamberlan',
        'Asolo', 'Tecnica', 'Dolomite', 'Aku', 'Bestard',
    ],

    # Invented/neologism brands that became successful
    'invented': [
        # Classic invented names
        'Kodak', 'Xerox', 'Exxon', 'Haagen', 'Acura',
        'Lexus', 'Infiniti', 'Altria', 'Diageo', 'Mondelez',
        # Tech invented names
        'Verizon', 'Accenture', 'Avaya', 'Agilent', 'Keysight',
        'Synopsys', 'Cadence', 'Splunk', 'Twilio', 'Datadog',
        'Snowflake', 'Databricks', 'Palantir', 'Confluent', 'Lacework',
        # Cybersecurity
        'Zscaler', 'Crowdstrike', 'Fortinet', 'Cloudflare', 'Fastly',
        'Akamai', 'Imperva', 'Qualys', 'Tenable', 'Sentinelone',
        'Cyberark', 'Okta', 'Sailpoint', 'Saviynt', 'Beyondtrust',
        # Fintech
        'Plaid', 'Brex', 'Ramp', 'Divvy', 'Navan',
        'Gusto', 'Rippling', 'Deel', 'Remote', 'Oyster',
    ],

    # German brands (strong in both DE and international markets)
    'german': [
        # Appliances
        'Braun', 'Miele', 'Liebherr', 'Blaupunkt', 'Grundig',
        'Siemens', 'Bosch', 'Neff', 'Gaggenau', 'Kueppersbusch',
        'Bauknecht', 'Aeg', 'Severin', 'Krups', 'Rowenta',
        # Audio
        'Sennheiser', 'Teufel', 'Beyerdynamic', 'Neumann', 'Adam',
        'Elac', 'Canton', 'Burmester', 'Mbl', 'Audionet',
        # Writing/Office
        'Staedtler', 'Faber', 'Pelikan', 'Lamy', 'Kaweco',
        'Montblanc', 'Leitz', 'Herlitz', 'Brunnen', 'Oxford',
        # Fashion/Lifestyle
        'Hugo', 'Joop', 'Escada', 'Bogner', 'Strenesse',
        'Closed', 'Drykorn', 'Marc', 'Windsor', 'Baldessarini',
        'Rimowa', 'Birkenstock', 'Bree', 'Aigner', 'Mcm',
        # Automotive
        'Porsche', 'Audi', 'Volkswagen', 'Opel', 'Daimler',
        'Bmw', 'Mercedes', 'Maybach', 'Smart', 'Mini',
    ],

    # Automotive & Mobility (for vehicle context relevance)
    'automotive': [
        # European
        'Volvo', 'Saab', 'Skoda', 'Seat', 'Cupra',
        'Peugeot', 'Renault', 'Citroen', 'Alpine', 'Dacia',
        'Fiat', 'Alfa', 'Lancia', 'Maserati', 'Ferrari',
        'Lamborghini', 'Pagani', 'Bugatti', 'Koenigsegg', 'Rimac',
        # Asian
        'Toyota', 'Honda', 'Nissan', 'Mazda', 'Subaru',
        'Mitsubishi', 'Suzuki', 'Isuzu', 'Daihatsu', 'Lexus',
        'Acura', 'Infiniti', 'Genesis', 'Hyundai', 'Kia',
        # Electric/New
        'Tesla', 'Rivian', 'Lucid', 'Polestar', 'Byton',
        'Nio', 'Xpeng', 'Byd', 'Geely', 'Zeekr',
        # Components
        'Bosch', 'Continental', 'Valeo', 'Denso', 'Aisin',
        'Magna', 'Aptiv', 'Lear', 'Autoliv', 'Faurecia',
    ],

    # Short punchy names (2-3 syllables, high memorability)
    'punchy': [
        'Apex', 'Bolt', 'Core', 'Dash', 'Edge',
        'Flux', 'Grid', 'Halo', 'Ion', 'Jolt',
        'Kite', 'Link', 'Mode', 'Node', 'Orb',
        'Pulse', 'Quest', 'Rift', 'Spark', 'Sync',
        'Trek', 'Unit', 'Vibe', 'Wave', 'Xero',
        'Zest', 'Zoom', 'Amp', 'Arc', 'Aero',
        'Beam', 'Blaze', 'Brisk', 'Craft', 'Crest',
        'Drift', 'Echo', 'Ember', 'Fuse', 'Gleam',
        'Grove', 'Haven', 'Hive', 'Jade', 'Kindle',
        'Lumen', 'Metro', 'Nimbus', 'Oasis', 'Peak',
        'Prism', 'Rapid', 'Ridge', 'Sage', 'Scout',
        'Slate', 'Stratus', 'Swift', 'Terra', 'Thrive',
        'Torque', 'Trail', 'Vapor', 'Vista', 'Zenith',
    ],
}

# Flatten corpus with optional category weighting
def get_weighted_corpus(weights: dict = None, max_name_length: int = 9) -> list[str]:
    """
    Get training corpus with optional category weighting.

    Args:
        weights: Category weights (higher = more influence)
        max_name_length: Filter out names longer than this (default: 9)
    """
    if weights is None:
        weights = {
            'camping': 1.2,      # Reduced - many long names
            'energy': 1.5,       # Good short names
            'tech': 1.0,
            'classic': 1.5,      # Many good short names (Nike, Uber, etc.)
            'classical': 2.0,    # Latin/Greek - mostly short, elegant
            'outdoor': 0.8,      # Reduced - many long names (Patagonia etc.)
            'invented': 0.8,     # Reduced - some too corporate
            'german': 1.0,
            'automotive': 0.8,   # Reduced - many long names
            'punchy': 3.0,       # HIGH weight - short memorable names
        }

    corpus = []
    for category, names in TRAINING_CORPUS.items():
        weight = weights.get(category, 1.0)
        # Filter by length - prefer shorter names
        filtered_names = [n for n in names if len(n) <= max_name_length]

        # Add names multiple times based on weight
        repeat = int(weight) + (1 if random.random() < (weight % 1) else 0)
        for _ in range(max(1, repeat)):
            corpus.extend(filtered_names)

    return corpus


# =============================================================================
# MARKOV CHAIN MODEL
# =============================================================================

@dataclass
class MarkovModel:
    """Character-level Markov chain model"""
    order: int
    transitions: dict = field(default_factory=dict)
    start_states: dict = field(default_factory=dict)
    char_frequencies: dict = field(default_factory=dict)

    # Special tokens
    START = '^'
    END = '$'

    def to_dict(self) -> dict:
        """Serialize model to dictionary"""
        return {
            'order': self.order,
            'transitions': dict(self.transitions),
            'start_states': dict(self.start_states),
            'char_frequencies': dict(self.char_frequencies),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MarkovModel':
        """Deserialize model from dictionary"""
        model = cls(order=data['order'])
        model.transitions = defaultdict(Counter, {
            k: Counter(v) for k, v in data['transitions'].items()
        })
        model.start_states = Counter(data['start_states'])
        model.char_frequencies = Counter(data['char_frequencies'])
        return model


class MarkovTrainer:
    """Trains Markov models on brand name corpora"""

    def __init__(self, order: int = 2):
        self.order = order

    def train(self, names: list[str]) -> MarkovModel:
        """Train a Markov model on a list of names"""
        model = MarkovModel(order=self.order)
        model.transitions = defaultdict(Counter)
        model.start_states = Counter()
        model.char_frequencies = Counter()

        for name in names:
            # Normalize: lowercase, strip whitespace
            name = name.lower().strip()
            if len(name) < 2:
                continue

            # Pad with start/end tokens
            padded = MarkovModel.START * self.order + name + MarkovModel.END

            # Learn start state (first n characters after padding)
            start_context = padded[self.order:self.order + self.order]
            model.start_states[start_context] += 1

            # Learn character frequencies
            for char in name:
                model.char_frequencies[char] += 1

            # Learn transitions
            for i in range(len(padded) - self.order):
                context = padded[i:i + self.order]
                next_char = padded[i + self.order]
                model.transitions[context][next_char] += 1

        return model

    def train_interpolated(self, names: list[str],
                           orders: list[int] = None) -> list[MarkovModel]:
        """Train multiple models for interpolation"""
        if orders is None:
            orders = [1, 2, 3]

        models = []
        for order in orders:
            trainer = MarkovTrainer(order=order)
            models.append(trainer.train(names))

        return models


class MarkovGenerator:
    """Generates names using trained Markov models"""

    def __init__(self,
                 models: list[MarkovModel],
                 interpolation_weights: list[float] = None):
        """
        Initialize generator with one or more models.

        Args:
            models: List of trained MarkovModels (different orders)
            interpolation_weights: Weights for each model (must sum to 1.0)
        """
        self.models = models

        if interpolation_weights is None:
            # Default: favor middle orders
            if len(models) == 1:
                self.weights = [1.0]
            elif len(models) == 2:
                self.weights = [0.3, 0.7]
            else:
                # e.g., for orders [1,2,3]: [0.15, 0.5, 0.35]
                self.weights = [0.15, 0.5, 0.35][:len(models)]
        else:
            self.weights = interpolation_weights

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def _sample_next_char(self,
                          context: str,
                          temperature: float = 1.0) -> Optional[str]:
        """
        Sample next character using interpolated probabilities.

        Args:
            context: Current context string
            temperature: Controls randomness (0.5=conservative, 1.5=creative)
        """
        # Collect probabilities from each model
        combined_probs = Counter()

        for model, weight in zip(self.models, self.weights):
            # Get context of appropriate length for this model
            model_context = context[-(model.order):] if len(context) >= model.order else (
                MarkovModel.START * (model.order - len(context)) + context
            )

            if model_context in model.transitions:
                for char, count in model.transitions[model_context].items():
                    # Apply temperature scaling
                    scaled_count = math.pow(count, 1.0 / temperature)
                    combined_probs[char] += weight * scaled_count

        if not combined_probs:
            return None

        # Sample from combined distribution
        chars = list(combined_probs.keys())
        probs = list(combined_probs.values())
        total = sum(probs)
        probs = [p / total for p in probs]

        return random.choices(chars, weights=probs)[0]

    def generate(self,
                 min_length: int = 3,
                 max_length: int = 12,
                 temperature: float = 1.0,
                 seed: str = None,
                 must_end_with_vowel: bool = False) -> Optional[str]:
        """
        Generate a single name.

        Args:
            min_length: Minimum name length
            max_length: Maximum name length
            temperature: Creativity (0.5=safe, 1.0=normal, 1.5=creative)
            seed: Optional starting string (e.g., semantic morpheme)
            must_end_with_vowel: Require name to end with a vowel

        Returns:
            Generated name or None if generation failed
        """
        # Initialize with seed or sampled start state
        if seed:
            result = seed.lower()
        else:
            # Sample start state from highest-order model
            main_model = self.models[-1]  # Highest order
            if not main_model.start_states:
                return None

            states = list(main_model.start_states.keys())
            counts = list(main_model.start_states.values())
            result = random.choices(states, weights=counts)[0].replace(
                MarkovModel.START, ''
            )

        # Generate until END token or max length
        attempts = 0
        max_attempts = max_length * 3

        while len(result) < max_length and attempts < max_attempts:
            attempts += 1

            next_char = self._sample_next_char(result, temperature)

            if next_char is None:
                break

            if next_char == MarkovModel.END:
                # Check length and ending constraints
                if len(result) >= min_length:
                    if must_end_with_vowel and result[-1] not in 'aeiou':
                        continue  # Try again
                    break
                else:
                    continue  # Too short, keep going

            result += next_char

        # Validate result
        if len(result) < min_length:
            return None

        if must_end_with_vowel and result[-1] not in 'aeiou':
            # Try to fix by adding a vowel
            result += random.choice(['a', 'i', 'o'])

        return result.capitalize()

    def generate_batch(self,
                       count: int,
                       temperature: float = 1.0,
                       seeds: list[str] = None,
                       **kwargs) -> list[str]:
        """
        Generate multiple unique names.

        Args:
            count: Number of names to generate
            temperature: Creativity level
            seeds: Optional list of seeds to use (cycles through)
            **kwargs: Additional arguments for generate()

        Returns:
            List of unique generated names
        """
        results = set()
        attempts = 0
        max_attempts = count * 20

        seed_idx = 0

        while len(results) < count and attempts < max_attempts:
            attempts += 1

            # Get seed if provided
            seed = None
            if seeds:
                seed = seeds[seed_idx % len(seeds)]
                seed_idx += 1

            name = self.generate(temperature=temperature, seed=seed, **kwargs)

            if name and name.lower() not in {n.lower() for n in results}:
                results.add(name)

        return list(results)


# =============================================================================
# HYBRID GENERATOR (Markov + Rules + Semantic)
# =============================================================================

class HybridMarkovGenerator:
    """
    Combines Markov chains with rule-based constraints and semantic seeding.

    This approach:
    1. Uses semantic morphemes as seeds for meaningful associations
    2. Applies Markov chains for natural-sounding continuations
    3. Filters with phonetic rules for DE/EN compatibility
    """

    # Semantic seeds organized by meaning
    SEMANTIC_SEEDS = {
        'energy': ['volt', 'amp', 'flux', 'watt', 'ohm', 'joule', 'erg',
                   'power', 'charge', 'pulse', 'spark', 'flow', 'current'],
        'travel': ['via', 'trek', 'road', 'path', 'way', 'route', 'trail',
                   'roam', 'wander', 'nomad', 'voyage', 'quest'],
        'nature': ['sol', 'lux', 'terra', 'aqua', 'aura', 'sky', 'wind',
                   'peak', 'mount', 'forest', 'river', 'ocean'],
        'tech': ['core', 'sync', 'link', 'hub', 'node', 'grid', 'net',
                 'data', 'cyber', 'digi', 'nano', 'micro'],
        'quality': ['prime', 'apex', 'max', 'ultra', 'pro', 'elite',
                    'premium', 'super', 'mega', 'hyper'],
    }

    # Good suffixes for brand names
    GOOD_SUFFIXES = ['ia', 'ix', 'on', 'ex', 'us', 'um', 'or', 'er',
                     'al', 'ar', 'an', 'en', 'is', 'os', 'ium']

    def __init__(self, corpus: list[str] = None, orders: list[int] = None):
        """
        Initialize hybrid generator.

        Args:
            corpus: Training corpus (uses default if None)
            orders: Markov chain orders (default: [2, 3])
        """
        if corpus is None:
            corpus = get_weighted_corpus()

        if orders is None:
            orders = [2, 3]

        # Train models
        trainer = MarkovTrainer()
        self.models = trainer.train_interpolated(corpus, orders)
        self.generator = MarkovGenerator(self.models)
        self.corpus_set = {name.lower() for name in corpus}

    def _is_valid_phonetically(self, name: str) -> bool:
        """Check if name follows phonetic rules for DE/EN pronounceability"""
        import re
        name_lower = name.lower()

        # === HARD FILTERS (instant reject) ===

        # No 4+ consecutive consonants
        if re.search(r'[bcdfghjklmnpqrstvwxz]{4,}', name_lower):
            return False

        # No 3+ consecutive vowels
        if re.search(r'[aeiou]{3,}', name_lower):
            return False

        # Must have at least one vowel
        if not any(c in name_lower for c in 'aeiou'):
            return False

        # Length check
        if len(name_lower) < 3 or len(name_lower) > 12:
            return False

        # === PROBLEMATIC PATTERNS ===

        # Awkward consonant clusters (hard for both DE and EN)
        bad_clusters = [
            'bv', 'bf', 'bw', 'bz', 'vb', 'fb', 'wb', 'zb',
            'pb', 'bp', 'pk', 'kp', 'gk', 'kg', 'dk', 'kd',
            'dt', 'td', 'dg', 'gd', 'db', 'bd',
            'fn', 'nf', 'vn', 'nv', 'wn', 'zn', 'nz',
            'mf', 'fm', 'mg', 'gm', 'mk', 'km',
            'pn', 'np', 'pm', 'mp' + 'h',  # mph is ok
            'cz', 'zc', 'xz', 'zx',
            'hh', 'jj', 'kk', 'qq', 'vv', 'ww', 'xx', 'yy', 'zz',
            'aa', 'ii', 'uu',
        ]
        for cluster in bad_clusters:
            if cluster in name_lower:
                return False

        # Awkward endings
        bad_endings = [
            'tb', 'db', 'kb', 'pb', 'gb',
            'tp', 'dp', 'kp', 'bp', 'gp',
            'tg', 'dg', 'kg', 'bg', 'pg',
            'bk', 'dk', 'gk', 'pk', 'tk',
            'bm', 'dm', 'gm', 'km', 'pm', 'tm',
            'bn', 'dn', 'gn', 'kn', 'pn', 'tn',
            'vh', 'wh', 'fh', 'bh', 'dh', 'gh', 'kh', 'ph', 'th',
            'fc', 'vc', 'wc', 'zc',
            'fj', 'vj', 'wj', 'zj', 'kj', 'gj', 'dj', 'bj', 'pj',
            'fq', 'vq', 'wq', 'zq', 'kq', 'gq', 'dq', 'bq', 'pq',
        ]
        if len(name_lower) >= 2 and name_lower[-2:] in bad_endings:
            return False

        # Awkward starts
        bad_starts = [
            'bv', 'bw', 'bz', 'dv', 'dw', 'dz',
            'gv', 'gw', 'gz', 'kv', 'kz',
            'pv', 'pw', 'pz', 'tv', 'tz',
            'fv', 'fw', 'fz', 'vf', 'vw', 'vz',
            'wv', 'wz', 'zv', 'zw',
            'sr', 'sd', 'sg', 'sf', 'sv', 'sz',
            'lr', 'ld', 'lg', 'lf', 'lv', 'lz',
            'nr', 'nd', 'ng', 'nf', 'nv', 'nz',
            'mr', 'md', 'mg', 'mf', 'mv', 'mz',
            'aa', 'ee', 'ii', 'oo', 'uu',
        ]
        if len(name_lower) >= 2 and name_lower[:2] in bad_starts:
            return False

        # === RATIO CHECKS ===

        consonants = sum(1 for c in name_lower if c in 'bcdfghjklmnpqrstvwxz')
        vowels = sum(1 for c in name_lower if c in 'aeiou')

        # Consonant-heavy names are hard to pronounce
        if vowels > 0 and consonants / vowels > 2.5:
            return False

        # Vowel-heavy names sound weak
        if consonants > 0 and vowels / consonants > 2:
            return False

        # === SYLLABLE CHECK ===

        # Count rough syllables (vowel groups)
        syllables = len(re.findall(r'[aeiou]+', name_lower))

        # Names should have 2-3 syllables ideally (1-4 acceptable)
        if syllables < 1 or syllables > 4:
            return False

        # Long names need more syllables
        if len(name_lower) >= 8 and syllables < 2:
            return False

        return True

    def _is_not_in_corpus(self, name: str) -> bool:
        """Check that we didn't just recreate a training example"""
        return name.lower() not in self.corpus_set

    def generate_seeded(self,
                        categories: list[str] = None,
                        temperature: float = 0.9,
                        min_length: int = 5,
                        max_length: int = 9) -> Optional[tuple[str, str]]:
        """
        Generate name starting with a semantic seed.

        Args:
            categories: Which semantic categories to use for seeds
            temperature: Creativity level
            min_length: Minimum total length
            max_length: Maximum total length

        Returns:
            Tuple of (name, seed_used) or None
        """
        if categories is None:
            categories = ['energy', 'travel', 'nature']

        # Collect all seeds from requested categories
        all_seeds = []
        for cat in categories:
            all_seeds.extend(self.SEMANTIC_SEEDS.get(cat, []))

        if not all_seeds:
            return None

        # Try multiple times
        for _ in range(50):
            seed = random.choice(all_seeds)

            # Generate continuation
            name = self.generator.generate(
                min_length=min_length,
                max_length=max_length,
                temperature=temperature,
                seed=seed
            )

            if (name and
                len(name) >= min_length and
                self._is_valid_phonetically(name) and
                self._is_not_in_corpus(name)):
                return (name, seed)

        return None

    def generate_with_suffix(self,
                             categories: list[str] = None,
                             temperature: float = 0.8) -> Optional[tuple[str, str]]:
        """
        Generate name with a curated suffix.

        Creates names like: [Markov-generated root] + [suffix]
        """
        if categories is None:
            categories = ['energy', 'travel']

        # Get a short seed (3-4 chars) from semantic pool
        all_seeds = []
        for cat in categories:
            all_seeds.extend(self.SEMANTIC_SEEDS.get(cat, []))

        short_seeds = [s for s in all_seeds if len(s) <= 4]

        for _ in range(50):
            # Either use seed or generate purely
            if short_seeds and random.random() < 0.6:
                root = random.choice(short_seeds)
            else:
                # Generate a short root
                root = self.generator.generate(
                    min_length=3,
                    max_length=5,
                    temperature=temperature
                )
                if not root:
                    continue
                root = root.lower()

            # Add suffix
            suffix = random.choice(self.GOOD_SUFFIXES)

            # Smooth the join
            if root[-1] in 'aeiou' and suffix[0] in 'aeiou':
                # Avoid double vowels
                name = root[:-1] + suffix
            elif root[-1] == suffix[0]:
                # Avoid double consonants
                name = root + suffix[1:]
            else:
                name = root + suffix

            name = name.capitalize()

            if (len(name) >= 4 and
                len(name) <= 10 and
                self._is_valid_phonetically(name) and
                self._is_not_in_corpus(name)):
                return (name, f"{root}+{suffix}")

        return None

    def generate_pure_markov(self,
                             temperature: float = 1.0,
                             min_length: int = 5,
                             max_length: int = 9) -> Optional[str]:
        """Generate purely from Markov chains (no seeding)"""
        for _ in range(50):
            name = self.generator.generate(
                min_length=min_length,
                max_length=max_length,
                temperature=temperature
            )

            if (name and
                self._is_valid_phonetically(name) and
                self._is_not_in_corpus(name)):
                return name

        return None

    def generate(self,
                 count: int = 30,
                 temperature: float = 0.9,
                 categories: list[str] = None,
                 strategy_weights: dict = None) -> list[tuple[str, str]]:
        """
        Generate multiple names using mixed strategies.

        Args:
            count: Number of names to generate
            temperature: Creativity level
            categories: Semantic categories to focus on
            strategy_weights: Weights for different generation strategies

        Returns:
            List of (name, generation_method) tuples
        """
        if strategy_weights is None:
            strategy_weights = {
                'seeded': 0.4,      # Semantic seed + Markov continuation
                'suffix': 0.35,     # Root + curated suffix
                'pure': 0.25,       # Pure Markov
            }

        results = []
        seen = set()
        attempts = 0
        max_attempts = count * 10

        strategies = list(strategy_weights.keys())
        weights = list(strategy_weights.values())

        while len(results) < count and attempts < max_attempts:
            attempts += 1

            strategy = random.choices(strategies, weights=weights)[0]
            name = None
            method = None

            if strategy == 'seeded':
                result = self.generate_seeded(
                    categories=categories,
                    temperature=temperature
                )
                if result:
                    name, seed = result
                    method = f"seeded({seed})"
            elif strategy == 'suffix':
                result = self.generate_with_suffix(
                    categories=categories,
                    temperature=temperature
                )
                if result:
                    name, components = result
                    method = f"suffix({components})"
            else:  # pure
                name = self.generate_pure_markov(temperature=temperature)
                if name:
                    method = "markov"

            if name and method:
                if name.lower() not in seen:
                    seen.add(name.lower())
                    results.append((name, method))

        return results


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_models(models: list[MarkovModel], filepath: str):
    """Save trained models to JSON file"""
    data = [m.to_dict() for m in models]
    Path(filepath).write_text(json.dumps(data, indent=2))


def load_models(filepath: str) -> list[MarkovModel]:
    """Load trained models from JSON file"""
    data = json.loads(Path(filepath).read_text())
    return [MarkovModel.from_dict(d) for d in data]


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate brand names using Markov chains'
    )
    parser.add_argument(
        '-n', '--count',
        type=int,
        default=20,
        help='Number of names to generate'
    )
    parser.add_argument(
        '-t', '--temperature',
        type=float,
        default=0.9,
        help='Creativity (0.5=conservative, 1.0=normal, 1.5=creative)'
    )
    parser.add_argument(
        '-c', '--categories',
        nargs='+',
        choices=['energy', 'travel', 'nature', 'tech', 'quality'],
        default=['energy', 'travel', 'nature'],
        help='Semantic categories for seeding'
    )
    parser.add_argument(
        '--pure-markov',
        action='store_true',
        help='Use only pure Markov generation (no semantic seeding)'
    )
    parser.add_argument(
        '--seed',
        type=str,
        help='Specific seed to start names with'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train and save models, do not generate'
    )
    parser.add_argument(
        '--save-models',
        type=str,
        help='Save trained models to this file'
    )
    parser.add_argument(
        '--load-models',
        type=str,
        help='Load pre-trained models from this file'
    )

    args = parser.parse_args()

    print("Markov Chain Brand Name Generator")
    print("=" * 50)

    # Get corpus
    corpus = get_weighted_corpus()
    print(f"Training corpus: {len(corpus)} names")

    # Train or load models
    if args.load_models:
        print(f"Loading models from {args.load_models}...")
        models = load_models(args.load_models)
    else:
        print("Training Markov models (orders 2, 3)...")
        trainer = MarkovTrainer()
        models = trainer.train_interpolated(corpus, [2, 3])

    if args.save_models:
        save_models(models, args.save_models)
        print(f"Models saved to {args.save_models}")

    if args.train_only:
        print("Training complete.")
        exit(0)

    print(f"\nGenerating {args.count} names (temperature={args.temperature})...")
    print("-" * 50)

    if args.pure_markov or args.seed:
        # Simple Markov generation
        generator = MarkovGenerator(models)

        if args.seed:
            names = generator.generate_batch(
                args.count,
                temperature=args.temperature,
                seeds=[args.seed]
            )
        else:
            names = generator.generate_batch(
                args.count,
                temperature=args.temperature
            )

        for i, name in enumerate(names, 1):
            print(f"{i:2}. {name}")

    else:
        # Hybrid generation
        hybrid = HybridMarkovGenerator()
        results = hybrid.generate(
            count=args.count,
            temperature=args.temperature,
            categories=args.categories
        )

        for i, (name, method) in enumerate(results, 1):
            print(f"{i:2}. {name:<15} [{method}]")

    print("-" * 50)
    print("Done!")
