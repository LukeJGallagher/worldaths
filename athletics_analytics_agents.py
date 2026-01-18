"""
Athletics Analytics Agents
Specialized knowledge and analysis for different athletics disciplines
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

###################################
# Configuration
###################################
SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')

###################################
# Athletics Knowledge Base
###################################

# Competition round types
ROUND_TYPES = {
    'Q': 'Qualification',
    'H': 'Heat',
    'SF': 'Semi-Final',
    'F': 'Final'
}

# Competition categories (World Athletics)
COMPETITION_CATEGORIES = {
    'OW': {'name': 'Olympic/World Championships', 'level': 1, 'points_mult': 1.0},
    'GL': {'name': 'Gold Label / Diamond League', 'level': 2, 'points_mult': 0.9},
    'A': {'name': 'Category A (Continental Champs)', 'level': 3, 'points_mult': 0.8},
    'B': {'name': 'Category B (National Champs)', 'level': 4, 'points_mult': 0.7},
    'C': {'name': 'Category C (International)', 'level': 5, 'points_mult': 0.6},
    'D': {'name': 'Category D (Regional)', 'level': 6, 'points_mult': 0.5},
    'E': {'name': 'Category E (Local)', 'level': 7, 'points_mult': 0.4},
    'F': {'name': 'Category F (Basic)', 'level': 8, 'points_mult': 0.3}
}

# Discipline knowledge
DISCIPLINE_KNOWLEDGE = {
    'sprints': {
        'events': ['100m', '200m', '400m'],
        'key_factors': [
            'Reaction time (crucial for 100m)',
            'Maximum velocity phase',
            'Speed endurance (400m)',
            'Lane draw (outer lanes disadvantage in 200m/400m curves)'
        ],
        'typical_progression': {
            '100m': {'heats_to_semis': 0.05, 'semis_to_final': 0.03},  # seconds typically faster
            '200m': {'heats_to_semis': 0.10, 'semis_to_final': 0.05},
            '400m': {'heats_to_semis': 0.30, 'semis_to_final': 0.20}
        },
        'altitude_effect': 'Beneficial (less air resistance)',
        'wind_legal_limit': 2.0,  # m/s
        'qualifying_rounds': {
            '100m': ['heats', 'semi-finals', 'final'],
            '200m': ['heats', 'semi-finals', 'final'],
            '400m': ['heats', 'semi-finals', 'final']
        },
        'advancement': {
            'auto_qualifiers': 3,  # first 3 in each heat
            'time_qualifiers': 'Fastest losers to fill remaining spots'
        }
    },
    'middle_distance': {
        'events': ['800m', '1500m'],
        'key_factors': [
            'Tactical awareness',
            'Kick/finishing speed',
            'Race positioning',
            'Pace judgment'
        ],
        'typical_splits': {
            '800m': {'first_400': 0.48, 'second_400': 0.52},  # percentage of total time
            '1500m': {'first_800': 0.52, 'last_700': 0.48}
        },
        'altitude_effect': 'Detrimental (reduced oxygen)',
        'race_tactics': [
            'Front running (aggressive pace)',
            'Sitting and kicking (save energy for sprint)',
            'Mid-pack positioning'
        ]
    },
    'long_distance': {
        'events': ['5000m', '10000m', 'marathon'],
        'key_factors': [
            'Aerobic capacity (VO2 max)',
            'Running economy',
            'Mental toughness',
            'Pacing strategy'
        ],
        'altitude_training': 'Critical for elite performance',
        'drafting_benefit': '3-6% energy saving',
        'marathon_specific': {
            'wall': 'Typically around 30-35km',
            'fueling': 'Carb loading, in-race nutrition',
            'optimal_temp': '10-12°C'
        }
    },
    'hurdles': {
        'events': ['110mh', '100mh', '400mh'],
        'key_factors': [
            'Hurdle technique',
            'Sprint speed between hurdles',
            'Rhythm maintenance',
            'Lead leg consistency'
        ],
        'technical_specs': {
            '110mh': {'height': 1.067, 'spacing': 9.14, 'hurdles': 10},
            '100mh': {'height': 0.838, 'spacing': 8.50, 'hurdles': 10},
            '400mh': {'height_men': 0.914, 'height_women': 0.762, 'hurdles': 10}
        },
        '400mh_stride_patterns': {
            'elite_men': '13 strides to first hurdle, 13-15 between',
            'elite_women': '15 strides to first hurdle, 15-17 between'
        }
    },
    'jumps': {
        'events': ['high-jump', 'long-jump', 'triple-jump', 'pole-vault'],
        'key_factors': {
            'high-jump': ['Approach speed', 'Takeoff angle', 'Bar clearance technique', 'Fosbury flop'],
            'long-jump': ['Approach speed', 'Takeoff accuracy', 'Flight technique', 'Landing'],
            'triple-jump': ['Hop-step-jump ratios', 'Phase balance', 'Speed maintenance'],
            'pole-vault': ['Grip height', 'Pole selection', 'Inversion technique']
        },
        'competition_format': {
            'attempts': 3,
            'final_attempts': 3,
            'advancement': 'Top 8 or 12 to final'
        },
        'wind_legal_limit': 2.0,  # m/s for long jump and triple jump
        'typical_ratios': {
            'triple-jump': {'hop': 0.35, 'step': 0.30, 'jump': 0.35}
        }
    },
    'throws': {
        'events': ['shot-put', 'discus-throw', 'hammer-throw', 'javelin-throw'],
        'key_factors': {
            'shot-put': ['Explosive power', 'Glide/Spin technique', 'Release angle'],
            'discus-throw': ['Rotational speed', 'Release timing', 'Wind reading'],
            'hammer-throw': ['Acceleration through turns', 'Balance', 'Release point'],
            'javelin-throw': ['Approach speed', 'Release angle', 'Aerodynamics']
        },
        'implement_weights': {
            'shot-put': {'men': 7.26, 'women': 4.0},
            'discus-throw': {'men': 2.0, 'women': 1.0},
            'hammer-throw': {'men': 7.26, 'women': 4.0},
            'javelin-throw': {'men': 0.8, 'women': 0.6}
        },
        'competition_format': {
            'qualification': 3,
            'final_attempts': 6,
            'cut_to_top_8': 'After 3 attempts'
        }
    },
    'combined_events': {
        'events': ['decathlon', 'heptathlon'],
        'decathlon_events': [
            '100m', 'long-jump', 'shot-put', 'high-jump', '400m',  # Day 1
            '110mh', 'discus-throw', 'pole-vault', 'javelin-throw', '1500m'  # Day 2
        ],
        'heptathlon_events': [
            '100mh', 'high-jump', 'shot-put', '200m',  # Day 1
            'long-jump', 'javelin-throw', '800m'  # Day 2
        ],
        'scoring': 'Points-based using IAAF scoring tables',
        'strategy': 'Balance strengths, minimize weaknesses',
        'key_factors': [
            'Versatility',
            'Recovery between events',
            'Mental resilience',
            'Energy management'
        ]
    }
}

# World Championship Finals Qualifying Standards (Top 8 marks historically)
FINALS_STANDARDS = {
    '100m': {
        'men': {'avg_top8': 10.02, 'slowest_qualifier': 10.08},
        'women': {'avg_top8': 10.98, 'slowest_qualifier': 11.05}
    },
    '200m': {
        'men': {'avg_top8': 20.15, 'slowest_qualifier': 20.25},
        'women': {'avg_top8': 22.35, 'slowest_qualifier': 22.50}
    },
    '400m': {
        'men': {'avg_top8': 44.85, 'slowest_qualifier': 45.10},
        'women': {'avg_top8': 50.25, 'slowest_qualifier': 50.60}
    },
    '800m': {
        'men': {'avg_top8': '1:44.50', 'slowest_qualifier': '1:45.20'},
        'women': {'avg_top8': '1:58.00', 'slowest_qualifier': '1:59.00'}
    },
    '1500m': {
        'men': {'avg_top8': '3:33.50', 'slowest_qualifier': '3:35.00'},
        'women': {'avg_top8': '4:02.00', 'slowest_qualifier': '4:04.00'}
    }
}


###################################
# Analytics Agent Classes
###################################

class BaseAthleticsAgent:
    """Base class for athletics analytics agents."""

    def __init__(self):
        self.sql_dir = SQL_DIR

    def load_ksa_data(self, gender='men'):
        """Load KSA athlete data from database."""
        db_path = os.path.join(self.sql_dir, f'ksa_modal_results_{gender}.db')
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql(f'SELECT * FROM ksa_modal_results_{gender}', conn)
                conn.close()
                return df
            except Exception as e:
                print(f"Warning: Could not load {gender} data: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def load_rankings(self, gender='men'):
        """Load world rankings data."""
        db_path = os.path.join(self.sql_dir, f'rankings_{gender}_all_events.db')
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            df = pd.read_sql(f'SELECT * FROM rankings_{gender}_all_events', conn)
            conn.close()
            return df
        return pd.DataFrame()


class SprintsAnalyzer(BaseAthleticsAgent):
    """Specialized analyzer for sprint events (100m, 200m, 400m)."""

    def __init__(self):
        super().__init__()
        self.knowledge = DISCIPLINE_KNOWLEDGE['sprints']
        self.events = self.knowledge['events']

    def analyze_round_progression(self, athlete_name: str, event: str) -> Dict:
        """Analyze athlete's performance progression through rounds."""
        df = self.load_ksa_data('men')
        if df.empty:
            df = self.load_ksa_data('women')

        if df.empty:
            return {'error': 'No data available'}

        # Filter for athlete and event
        athlete_data = df[
            (df['Athlete'].str.contains(athlete_name, case=False, na=False)) &
            (df['Event Type'] == event)
        ].copy()

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name} in {event}'}

        # Group by round type
        results = {}
        for round_type in ['H', 'SF', 'F']:
            round_data = athlete_data[athlete_data['Type'] == round_type]
            if not round_data.empty:
                results[ROUND_TYPES.get(round_type, round_type)] = {
                    'count': len(round_data),
                    'best': round_data['Result'].min(),
                    'avg': round_data['Result'].mean() if round_data['Result'].dtype != 'object' else 'N/A'
                }

        return {
            'athlete': athlete_name,
            'event': event,
            'rounds': results,
            'knowledge': {
                'typical_improvement': self.knowledge['typical_progression'].get(event, {}),
                'wind_limit': self.knowledge['wind_legal_limit']
            }
        }

    def get_finals_readiness(self, athlete_name: str, event: str) -> Dict:
        """Assess if athlete is ready for major championship finals."""
        df = self.load_ksa_data('men')
        if df.empty:
            df = self.load_ksa_data('women')

        athlete_data = df[
            (df['Athlete'].str.contains(athlete_name, case=False, na=False)) &
            (df['Event Type'] == event)
        ]

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        # Get best result
        best_result = athlete_data['Result'].min()

        # Compare to finals standards
        standards = FINALS_STANDARDS.get(event, {})
        gender = 'men' if 'men' in athlete_data['Gender'].values else 'women'

        if gender in standards:
            slowest_qualifier = standards[gender]['slowest_qualifier']
            gap = float(best_result) - float(slowest_qualifier) if isinstance(best_result, (int, float)) else None

            return {
                'athlete': athlete_name,
                'event': event,
                'pb': best_result,
                'finals_standard': slowest_qualifier,
                'gap': f"{gap:+.2f}s" if gap else 'N/A',
                'assessment': 'Finals contender' if gap and gap <= 0 else 'Needs improvement' if gap else 'Cannot assess'
            }

        return {'athlete': athlete_name, 'event': event, 'pb': best_result, 'assessment': 'No standards data'}


class MiddleDistanceAnalyzer(BaseAthleticsAgent):
    """Specialized analyzer for middle distance events (800m, 1500m)."""

    def __init__(self):
        super().__init__()
        self.knowledge = DISCIPLINE_KNOWLEDGE['middle_distance']
        self.events = self.knowledge['events']

    def analyze_tactical_performance(self, athlete_name: str, event: str) -> Dict:
        """Analyze tactical aspects of middle distance racing."""
        df = self.load_ksa_data('men')
        if df.empty:
            df = self.load_ksa_data('women')

        athlete_data = df[
            (df['Athlete'].str.contains(athlete_name, case=False, na=False)) &
            (df['Event Type'] == event)
        ]

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        # Analyze placement patterns
        finals = athlete_data[athlete_data['Type'] == 'F']

        placements = finals['Pl.'].value_counts().to_dict() if not finals.empty else {}

        return {
            'athlete': athlete_name,
            'event': event,
            'finals_competed': len(finals),
            'placement_distribution': placements,
            'tactical_advice': self.knowledge['race_tactics'],
            'typical_splits': self.knowledge['typical_splits'].get(event, {})
        }


class JumpsAnalyzer(BaseAthleticsAgent):
    """Specialized analyzer for jumping events."""

    def __init__(self):
        super().__init__()
        self.knowledge = DISCIPLINE_KNOWLEDGE['jumps']
        self.events = self.knowledge['events']

    def analyze_competition_performance(self, athlete_name: str, event: str) -> Dict:
        """Analyze jumping competition performance."""
        df = self.load_ksa_data('men')
        if df.empty:
            df = self.load_ksa_data('women')

        athlete_data = df[
            (df['Athlete'].str.contains(athlete_name, case=False, na=False)) &
            (df['Event Type'].str.contains(event.replace('-', ''), case=False, na=False))
        ]

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        # Analyze by competition category
        by_category = athlete_data.groupby('Cat.').agg({
            'Result': ['count', 'max'],
            'Pl.': 'min'
        }).reset_index()

        return {
            'athlete': athlete_name,
            'event': event,
            'competitions': len(athlete_data),
            'by_category': by_category.to_dict() if not by_category.empty else {},
            'key_factors': self.knowledge['key_factors'].get(event, []),
            'competition_format': self.knowledge['competition_format']
        }


class ThrowsAnalyzer(BaseAthleticsAgent):
    """Specialized analyzer for throwing events."""

    def __init__(self):
        super().__init__()
        self.knowledge = DISCIPLINE_KNOWLEDGE['throws']
        self.events = self.knowledge['events']

    def analyze_throwing_performance(self, athlete_name: str, event: str) -> Dict:
        """Analyze throwing competition performance."""
        df = self.load_ksa_data('men')
        if df.empty:
            df = self.load_ksa_data('women')

        athlete_data = df[
            (df['Athlete'].str.contains(athlete_name, case=False, na=False)) &
            (df['Event Type'].str.contains(event.replace('-', ' '), case=False, na=False))
        ]

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        return {
            'athlete': athlete_name,
            'event': event,
            'competitions': len(athlete_data),
            'best_result': athlete_data['Result'].max() if not athlete_data.empty else 'N/A',
            'implement_weight': self.knowledge['implement_weights'].get(event, {}),
            'key_factors': self.knowledge['key_factors'].get(event, [])
        }


class RoundAnalyzer(BaseAthleticsAgent):
    """Analyzer for heats, semis, and finals progression."""

    def __init__(self):
        super().__init__()

    def get_round_statistics(self, event: str = None) -> Dict:
        """Get statistics for different rounds."""
        df_men = self.load_ksa_data('men')
        df_women = self.load_ksa_data('women')
        df = pd.concat([df_men, df_women]) if not df_men.empty else df_women

        if df.empty:
            return {'error': 'No data available'}

        if event:
            df = df[df['Event Type'] == event]

        stats = {}
        for round_code, round_name in ROUND_TYPES.items():
            round_data = df[df['Type'] == round_code]
            if not round_data.empty:
                stats[round_name] = {
                    'total_performances': len(round_data),
                    'unique_athletes': round_data['Athlete'].nunique(),
                    'events': round_data['Event Type'].unique().tolist(),
                    'avg_place': round_data['Pl.'].str.extract(r'(\d+)')[0].astype(float).mean() if 'Pl.' in round_data.columns else 'N/A'
                }

        return stats

    def analyze_heat_to_final_progression(self, athlete_name: str, event: str) -> Dict:
        """Analyze how athlete progresses from heats to finals."""
        df = self.load_ksa_data('men')
        if df.empty:
            df = self.load_ksa_data('women')

        athlete_data = df[
            (df['Athlete'].str.contains(athlete_name, case=False, na=False)) &
            (df['Event Type'] == event)
        ]

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        progression = {}
        for round_code in ['H', 'SF', 'F']:
            round_data = athlete_data[athlete_data['Type'] == round_code]
            if not round_data.empty:
                progression[ROUND_TYPES[round_code]] = {
                    'count': len(round_data),
                    'results': round_data['Result'].tolist(),
                    'best': round_data['Result'].min(),
                    'competitions': round_data['Competition'].unique().tolist()
                }

        # Calculate progression rate
        heat_count = len(athlete_data[athlete_data['Type'] == 'H'])
        semi_count = len(athlete_data[athlete_data['Type'] == 'SF'])
        final_count = len(athlete_data[athlete_data['Type'] == 'F'])

        return {
            'athlete': athlete_name,
            'event': event,
            'progression': progression,
            'conversion_rates': {
                'heats_run': heat_count,
                'semis_reached': semi_count,
                'finals_reached': final_count,
                'heat_to_semi_rate': f"{semi_count/heat_count*100:.1f}%" if heat_count > 0 else 'N/A',
                'semi_to_final_rate': f"{final_count/semi_count*100:.1f}%" if semi_count > 0 else 'N/A'
            }
        }


class CompetitionCategoryAnalyzer(BaseAthleticsAgent):
    """Analyzer for competition category performance."""

    def __init__(self):
        super().__init__()
        self.categories = COMPETITION_CATEGORIES

    def analyze_by_competition_level(self, athlete_name: str = None) -> Dict:
        """Analyze performance across competition levels."""
        df = self.load_ksa_data('men')
        if df.empty:
            df = self.load_ksa_data('women')

        if df.empty:
            return {'error': 'No data available'}

        if athlete_name:
            df = df[df['Athlete'].str.contains(athlete_name, case=False, na=False)]

        results = {}
        for cat_code, cat_info in self.categories.items():
            cat_data = df[df['Cat.'] == cat_code]
            if not cat_data.empty:
                # Get top results safely
                top_results = []
                if 'Result' in cat_data.columns:
                    result_data = cat_data[cat_data['Result'].notna()]
                    if not result_data.empty:
                        top_results = result_data.head(3)['Result'].tolist()

                results[cat_code] = {
                    'name': cat_info['name'],
                    'level': cat_info['level'],
                    'performances': len(cat_data),
                    'unique_events': cat_data['Event Type'].nunique(),
                    'top_results': top_results
                }

        return {
            'athlete': athlete_name or 'All KSA Athletes',
            'by_category': results,
            'recommendation': self._get_competition_recommendation(results)
        }

    def _get_competition_recommendation(self, results: Dict) -> str:
        """Generate recommendation based on competition level performance."""
        if not results:
            return "Need more competition data"

        high_level = sum(1 for cat in results if results[cat]['level'] <= 3)

        if high_level >= 3:
            return "Athlete competing regularly at high level - maintain championship focus"
        elif high_level >= 1:
            return "Some high-level experience - increase A/B category competitions"
        else:
            return "Focus on gaining experience at higher competition levels"


###################################
# Major Games Analyzer
###################################

# Major games categorization
MAJOR_GAMES = {
    'Olympic': ['Olympic Games', 'XXXIII Olympic', 'XXXII Olympic', 'XXXI Olympic'],
    'World Championships': ['World Athletics Championships', 'World Championships'],
    'World U20': ['World Athletics U20', 'World U20 Championships'],
    'Asian Games': ['Asian Games'],
    'Asian Championships': ['Asian Athletics Championships', 'Asian Indoor'],
    'West Asian': ['West Asian Championships'],
    'Arab Championships': ['Arab Athletics', 'Arab U18', 'Arab U20', 'Arab U23'],
    'GCC': ['GCC Youth Games', 'GCC Championships']
}


class MajorGamesAnalyzer(BaseAthleticsAgent):
    """Analyzer for major games performance."""

    def __init__(self):
        super().__init__()
        self.major_games = MAJOR_GAMES
        self.profiles_db = os.path.join(self.sql_dir, 'ksa_athlete_profiles.db')

    def load_profiles_results(self):
        """Load results from athlete profiles database."""
        if os.path.exists(self.profiles_db):
            try:
                conn = sqlite3.connect(self.profiles_db)
                df = pd.read_sql("""
                    SELECT r.*, a.full_name, a.primary_event
                    FROM athlete_results r
                    JOIN ksa_athletes a ON r.athlete_id = a.athlete_id
                """, conn)
                conn.close()
                return df
            except Exception as e:
                print(f"Warning: Could not load profiles data: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def get_major_games_summary(self) -> Dict:
        """Get summary of KSA performance at major games."""
        df = self.load_profiles_results()
        if df.empty:
            return {'error': 'No data available'}

        summary = {}
        for game_type in self.major_games.keys():
            game_data = df[df['game_category'] == game_type]
            if not game_data.empty:
                summary[game_type] = {
                    'total_performances': len(game_data),
                    'unique_athletes': game_data['full_name'].nunique(),
                    'events': game_data['event_name'].unique().tolist(),
                    'best_placements': game_data[game_data['place'].notna()]['place'].value_counts().head(5).to_dict()
                }

        return summary

    def analyze_athlete_major_games(self, athlete_name: str) -> Dict:
        """Analyze athlete's major games performance."""
        df = self.load_profiles_results()
        if df.empty:
            return {'error': 'No data available'}

        athlete_data = df[df['full_name'].str.contains(athlete_name, case=False, na=False)]
        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        results = {
            'athlete': athlete_name,
            'total_major_performances': 0,
            'by_game_type': {},
            'highlights': []
        }

        for game_type in self.major_games.keys():
            game_data = athlete_data[athlete_data['game_category'] == game_type]
            if not game_data.empty:
                results['total_major_performances'] += len(game_data)
                results['by_game_type'][game_type] = {
                    'count': len(game_data),
                    'events': game_data['event_name'].unique().tolist(),
                    'results': game_data[['event_name', 'result_value', 'place', 'competition_name']].to_dict('records')
                }

                # Find best placement
                best_place = game_data[game_data['place'].notna()]['place'].min()
                if best_place and best_place <= 8:
                    results['highlights'].append({
                        'game': game_type,
                        'place': int(best_place),
                        'event': game_data[game_data['place'] == best_place].iloc[0]['event_name']
                    })

        return results

    def get_olympic_performers(self) -> Dict:
        """Get all KSA athletes who competed at Olympics."""
        df = self.load_profiles_results()
        if df.empty:
            return {'error': 'No data available'}

        olympic_data = df[df['game_category'] == 'Olympic']
        if olympic_data.empty:
            return {'message': 'No Olympic performances found'}

        return {
            'total_performances': len(olympic_data),
            'athletes': olympic_data.groupby('full_name').agg({
                'event_name': list,
                'result_value': list,
                'place': 'min',
                'competition_name': 'first'
            }).to_dict('index')
        }

    def get_finals_appearances(self) -> Dict:
        """Get KSA athletes who made finals at major games."""
        df = self.load_profiles_results()
        if df.empty:
            return {'error': 'No data available'}

        # Filter for finals (round = 'F') at major games
        major_game_types = list(self.major_games.keys())
        finals_data = df[(df['round'] == 'F') & (df['game_category'].isin(major_game_types))]

        if finals_data.empty:
            return {'message': 'No finals appearances at major games'}

        return {
            'total_finals': len(finals_data),
            'by_athlete': finals_data.groupby('full_name').agg({
                'game_category': list,
                'event_name': list,
                'result_value': list,
                'place': list
            }).to_dict('index')
        }


###################################
# Main Analytics Interface
###################################

class AthleticsAnalytics:
    """Main interface for athletics analytics."""

    def __init__(self):
        self.sprints = SprintsAnalyzer()
        self.middle_distance = MiddleDistanceAnalyzer()
        self.jumps = JumpsAnalyzer()
        self.throws = ThrowsAnalyzer()
        self.rounds = RoundAnalyzer()
        self.competition = CompetitionCategoryAnalyzer()
        self.major_games = MajorGamesAnalyzer()

    def get_discipline_knowledge(self, discipline: str) -> Dict:
        """Get knowledge base for a specific discipline."""
        return DISCIPLINE_KNOWLEDGE.get(discipline, {'error': 'Unknown discipline'})

    def get_finals_standards(self, event: str) -> Dict:
        """Get finals qualifying standards for an event."""
        return FINALS_STANDARDS.get(event, {'error': 'No standards for this event'})

    def analyze_athlete(self, athlete_name: str, event: str) -> Dict:
        """Comprehensive athlete analysis."""
        # Determine discipline
        if event in ['100m', '200m', '400m']:
            analyzer = self.sprints
        elif event in ['800m', '1500m']:
            analyzer = self.middle_distance
        elif 'jump' in event or 'vault' in event:
            analyzer = self.jumps
        elif 'throw' in event or 'put' in event:
            analyzer = self.throws
        else:
            analyzer = self.rounds

        results = {
            'athlete': athlete_name,
            'event': event,
            'round_progression': self.rounds.analyze_heat_to_final_progression(athlete_name, event),
            'competition_levels': self.competition.analyze_by_competition_level(athlete_name)
        }

        # Add discipline-specific analysis
        if hasattr(analyzer, 'analyze_round_progression'):
            results['discipline_analysis'] = analyzer.analyze_round_progression(athlete_name, event)

        return results


# Quick test function
def test_analytics():
    """Test the analytics agents."""
    analytics = AthleticsAnalytics()

    print("=" * 60)
    print("Athletics Analytics Test")
    print("=" * 60)

    # Test round statistics
    print("\n1. Round Statistics:")
    round_stats = analytics.rounds.get_round_statistics('100m')
    for round_name, stats in round_stats.items():
        print(f"  {round_name}: {stats.get('total_performances', 0)} performances")

    # Test athlete analysis
    print("\n2. Athlete Analysis (Nasser Mohammed - 100m):")
    analysis = analytics.analyze_athlete('Nasser', '100m')
    if 'error' not in analysis.get('round_progression', {}):
        prog = analysis['round_progression']
        print(f"  Finals reached: {prog.get('conversion_rates', {}).get('finals_reached', 0)}")

    # Test discipline knowledge
    print("\n3. Sprints Knowledge:")
    knowledge = analytics.get_discipline_knowledge('sprints')
    print(f"  Events: {knowledge.get('events', [])}")
    print(f"  Wind limit: {knowledge.get('wind_legal_limit', 'N/A')} m/s")

    # Test major games analysis
    print("\n4. Major Games Summary:")
    major_summary = analytics.major_games.get_major_games_summary()
    for game_type, data in major_summary.items():
        if isinstance(data, dict):
            print(f"  {game_type}: {data.get('total_performances', 0)} performances, {data.get('unique_athletes', 0)} athletes")

    # Test Olympic performers
    print("\n5. Olympic Performers:")
    olympic = analytics.major_games.get_olympic_performers()
    if 'athletes' in olympic:
        print(f"  Total Olympic performances: {olympic.get('total_performances', 0)}")
        for athlete, data in list(olympic.get('athletes', {}).items())[:3]:
            print(f"    - {athlete}: {data.get('event_name', [])}")

    # Test finals appearances
    print("\n6. Finals at Major Games:")
    finals = analytics.major_games.get_finals_appearances()
    if 'by_athlete' in finals:
        print(f"  Total finals appearances: {finals.get('total_finals', 0)}")
        for athlete, data in list(finals.get('by_athlete', {}).items())[:5]:
            print(f"    - {athlete}: {list(set(data.get('game_category', [])))}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == '__main__':
    test_analytics()
