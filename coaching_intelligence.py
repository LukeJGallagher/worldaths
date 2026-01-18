"""
Coaching Intelligence System
Head Coach Agent + Head of Data Analytics Integration

Provides deep tactical insights for athlete development decisions.
Combines global performance standards with discipline-specific knowledge.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Team Saudi Brand Colors
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'

SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'world_athletics_scraperv2', 'data')

# =============================================================================
# DISCIPLINE-SPECIFIC COACHING KNOWLEDGE (Deep Domain Expertise)
# =============================================================================

COACHING_INSIGHTS = {
    'sprints': {
        'events': ['100m', '200m', '400m', '100-metres', '200-metres', '400-metres'],
        'key_performance_factors': {
            '100m': {
                'reaction_time': {'elite_range': (0.12, 0.15), 'weight': 0.15},
                'acceleration_0_30m': {'elite_range': (3.75, 3.90), 'weight': 0.30},
                'max_velocity_60_80m': {'elite_range': (11.5, 12.0), 'weight': 0.35},  # m/s
                'deceleration_control': {'elite_range': (0.02, 0.05), 'weight': 0.20}  # % slowdown
            },
            '200m': {
                'curve_efficiency': {'elite_range': (0.97, 0.99), 'weight': 0.25},
                'speed_endurance': {'elite_range': (0.95, 0.98), 'weight': 0.40},
                'max_velocity': {'elite_range': (11.3, 11.8), 'weight': 0.35}
            },
            '400m': {
                'first_200m_split': {'elite_pct': (48.5, 50.0), 'weight': 0.30},
                'speed_endurance_200_300m': {'elite_range': (0.92, 0.96), 'weight': 0.35},
                'lactate_tolerance_300_400m': {'elite_range': (0.88, 0.93), 'weight': 0.35}
            }
        },
        'training_priorities': {
            'gap_to_medal > 0.5s': ['Basic speed development', 'Technique foundation', 'General conditioning'],
            'gap_to_medal 0.3-0.5s': ['Speed endurance', 'Race-specific training', 'Block starts'],
            'gap_to_medal 0.1-0.3s': ['Reaction time drills', 'Competition simulation', 'Mental preparation'],
            'gap_to_medal < 0.1s': ['Marginal gains', 'Altitude camps', 'Championship peaking']
        },
        'round_progression': {
            '100m': {'heats_to_semis': 0.03, 'semis_to_final': 0.02, 'pct_save_heats': 0.97},
            '200m': {'heats_to_semis': 0.08, 'semis_to_final': 0.05, 'pct_save_heats': 0.96},
            '400m': {'heats_to_semis': 0.25, 'semis_to_final': 0.15, 'pct_save_heats': 0.98}
        }
    },
    'middle_distance': {
        'events': ['800m', '1500m', '800-metres', '1500-metres'],
        'tactical_profiles': {
            'front_runner': {
                'description': 'Pushes pace from gun, dares others to follow',
                'ideal_for': 'Athletes with strong speed endurance, weaker kick',
                'risk': 'High - can be caught in final 100m if pace drops'
            },
            'sit_and_kick': {
                'description': 'Stays in pack, unleashes finishing speed',
                'ideal_for': 'Athletes with 400m speed background',
                'risk': 'Medium - can get boxed in, tactical fouls'
            },
            'mid_pack_controller': {
                'description': 'Stays in contact, surges at 300m to go',
                'ideal_for': 'Well-rounded athletes with race intelligence',
                'risk': 'Low - versatile but requires experience'
            }
        },
        'split_analysis': {
            '800m': {
                'optimal_first_400': {'pct_range': (49.5, 51.5), 'elite_example': '50.5/52.5 for 1:43'},
                'even_pacing_benefit': '1-2 seconds over positive split'
            },
            '1500m': {
                'optimal_800m_split': {'pct_range': (51, 53), 'elite_example': '1:52 for 3:30 finish'},
                'last_300m_kick': {'elite_range': (38.5, 41.0)}
            }
        },
        'training_priorities': {
            'gap_to_medal > 5s': ['Aerobic base', 'VO2 max development', 'Running economy'],
            'gap_to_medal 2-5s': ['Speed endurance', 'Race pace training', 'Altitude blocks'],
            'gap_to_medal < 2s': ['Tactical training', 'Championship simulation', 'Finishing speed']
        }
    },
    'long_distance': {
        'events': ['5000m', '10000m', 'marathon', '5000-metres', '10000-metres'],
        'performance_factors': {
            'vo2_max': {'elite_range': (75, 85), 'unit': 'ml/kg/min', 'trainable': True},
            'running_economy': {'elite_range': (180, 210), 'unit': 'ml/kg/km', 'trainable': True},
            'lactate_threshold': {'elite_pct_vo2max': (85, 92), 'trainable': True},
            'fractional_utilization': {'elite_pct': (88, 95), 'trainable': True}
        },
        'marathon_specific': {
            'glycogen_depletion_wall': {'typical_km': (30, 35), 'prevention': 'Fueling strategy'},
            'optimal_temperature': {'range_c': (10, 15), 'impact_per_degree': '0:30-1:00 per 5°C above'},
            'drafting_benefit': {'pct_energy_saved': (3, 6)}
        },
        'training_priorities': {
            'gap_to_medal > 3min': ['Base mileage building', 'Aerobic conditioning', 'Core strength'],
            'gap_to_medal 1-3min': ['Threshold training', 'Long runs', 'Race pace work'],
            'gap_to_medal < 1min': ['Specific race preparation', 'Altitude camps', 'Peaking protocols']
        }
    },
    'hurdles': {
        'events': ['110mH', '100mH', '400mH', '110-metres-hurdles', '100-metres-hurdles', '400-metres-hurdles'],
        'technical_factors': {
            '110mH': {
                'hurdle_height': 1.067,
                'inter_hurdle_distance': 9.14,
                'elite_stride_pattern': 3,
                'key_metrics': ['Lead leg speed', 'Trail leg efficiency', 'Hurdle clearance height']
            },
            '100mH': {
                'hurdle_height': 0.838,
                'inter_hurdle_distance': 8.50,
                'elite_stride_pattern': 3,
                'key_metrics': ['Rhythm maintenance', 'Attack angle', 'Ground contact time']
            },
            '400mH': {
                'hurdle_height': 0.914,
                'inter_hurdle_distance': 35.0,
                'elite_stride_patterns': {'first_half': (13, 15), 'second_half': (15, 17)},
                'key_metrics': ['Stride pattern consistency', 'Lactate tolerance', 'Lead leg switching']
            }
        },
        'training_priorities': {
            'gap_to_medal > 0.5s': ['Sprint speed development', 'Hurdle technique foundation', 'Flexibility'],
            'gap_to_medal 0.2-0.5s': ['Race rhythm', 'Specific endurance', 'Competition simulation'],
            'gap_to_medal < 0.2s': ['Marginal technique gains', 'Mental preparation', 'Race tactics']
        }
    },
    'jumps': {
        'events': ['high-jump', 'long-jump', 'triple-jump', 'pole-vault'],
        'biomechanical_factors': {
            'high-jump': {
                'approach_speed': {'elite_range': (7.5, 8.5), 'unit': 'm/s'},
                'takeoff_angle': {'elite_range': (40, 50), 'unit': 'degrees'},
                'bar_clearance': {'elite_margin': (0.03, 0.08), 'unit': 'm above COM'}
            },
            'long-jump': {
                'approach_speed': {'elite_range': (10.5, 11.2), 'unit': 'm/s'},
                'takeoff_angle': {'elite_range': (18, 24), 'unit': 'degrees'},
                'flight_technique': ['Hang', 'Hitch-kick', 'Sail']
            },
            'triple-jump': {
                'phase_ratios': {'hop': 35, 'step': 30, 'jump': 35},
                'approach_speed': {'elite_range': (10.0, 10.8), 'unit': 'm/s'},
                'phase_balance': 'Maintain rhythm, avoid overemphasis on hop'
            },
            'pole-vault': {
                'approach_speed': {'elite_range': (9.5, 10.5), 'unit': 'm/s'},
                'grip_height': {'elite_range': (4.80, 5.20), 'unit': 'm'},
                'pole_stiffness': 'Matched to approach speed and body weight'
            }
        },
        'training_priorities': {
            'gap_to_medal > 0.50m': ['Approach speed development', 'Basic technique', 'Strength foundation'],
            'gap_to_medal 0.20-0.50m': ['Technique refinement', 'Specific drills', 'Competition experience'],
            'gap_to_medal < 0.20m': ['Fine-tuning', 'Mental skills', 'Championship preparation']
        }
    },
    'throws': {
        'events': ['shot-put', 'discus-throw', 'hammer-throw', 'javelin-throw'],
        'implement_specs': {
            'shot-put': {'men': 7.26, 'women': 4.0, 'unit': 'kg'},
            'discus-throw': {'men': 2.0, 'women': 1.0, 'unit': 'kg'},
            'hammer-throw': {'men': 7.26, 'women': 4.0, 'unit': 'kg'},
            'javelin-throw': {'men': 0.8, 'women': 0.6, 'unit': 'kg'}
        },
        'performance_factors': {
            'release_velocity': {'primary_factor': True, 'accounts_for': '70-80% of distance'},
            'release_angle': {'optimal_range': (34, 42), 'varies_by_event': True},
            'release_height': {'benefit_per_10cm': '0.5-1.0m distance'}
        },
        'training_priorities': {
            'gap_to_medal > 5m': ['Strength development', 'Basic technique', 'Implement feel'],
            'gap_to_medal 2-5m': ['Power development', 'Technique refinement', 'Release mechanics'],
            'gap_to_medal < 2m': ['Competition simulation', 'Mental preparation', 'Peaking']
        }
    }
}


# =============================================================================
# PERFORMANCE GAP ANALYZER
# =============================================================================

@dataclass
class PerformanceGap:
    """Structured representation of gap to target."""
    athlete_name: str
    event: str
    current_pb: float
    target_standard: str  # 'gold', 'silver', 'bronze', 'finals'
    target_mark: float
    gap: float
    gap_percentage: float
    training_priority: str
    estimated_improvement_potential: str
    is_field_event: bool


class CoachingIntelligence:
    """Head Coach decision support system."""

    def __init__(self):
        self.profiles_db = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')
        self.wittw_db = os.path.join(SQL_DIR, 'what_it_takes_to_win.db')
        self.data_path = os.path.join(DATA_DIR, 'db_cleaned.csv')

    def _is_field_event(self, event: str) -> bool:
        """Check if event is field (higher is better) vs track (lower is better)."""
        field_keywords = ['jump', 'vault', 'put', 'throw', 'discus', 'javelin', 'hammer', 'decathlon', 'heptathlon']
        return any(kw in event.lower() for kw in field_keywords)

    def _parse_mark(self, mark_str: str, is_field: bool) -> Optional[float]:
        """Parse performance mark to comparable numeric."""
        if pd.isna(mark_str) or mark_str == '':
            return None

        mark_str = str(mark_str).strip()

        # Handle invalid marks
        if any(x in mark_str.upper() for x in ['DNF', 'DNS', 'DQ', 'NM', '-']):
            return None

        try:
            # Field events - just parse as float
            if is_field:
                return float(mark_str.replace('m', '').replace('pts', '').strip())

            # Track events - handle time formats
            if ':' in mark_str:
                parts = mark_str.split(':')
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            return float(mark_str)
        except ValueError:
            return None

    def _format_mark(self, seconds: float, event: str) -> str:
        """Format numeric mark back to readable string."""
        if seconds is None:
            return 'N/A'

        is_field = self._is_field_event(event)

        if is_field:
            if 'decathlon' in event.lower() or 'heptathlon' in event.lower():
                return f"{int(seconds)} pts"
            return f"{seconds:.2f}m"

        # Track events
        if seconds >= 3600:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h}:{m:02d}:{s:05.2f}"
        elif seconds >= 60:
            m = int(seconds // 60)
            s = seconds % 60
            return f"{m}:{s:05.2f}"
        else:
            return f"{seconds:.2f}"

    def _get_discipline_for_event(self, event: str) -> str:
        """Map event to discipline category."""
        event_lower = event.lower()
        for discipline, info in COACHING_INSIGHTS.items():
            for e in info['events']:
                if e.lower() in event_lower or event_lower in e.lower():
                    return discipline
        return 'unknown'

    def analyze_athlete_gap(self, athlete_name: str, event: str = None) -> List[PerformanceGap]:
        """Analyze performance gaps for an athlete."""
        gaps = []

        if not os.path.exists(self.profiles_db):
            return gaps

        conn = sqlite3.connect(self.profiles_db)

        # Get athlete's PBs
        query = """
            SELECT a.full_name, p.event_name, p.pb_result
            FROM ksa_athletes a
            JOIN athlete_pbs p ON a.athlete_id = p.athlete_id
            WHERE a.full_name = ?
        """
        params = [athlete_name]
        if event:
            query += " AND p.event_name LIKE ?"
            params.append(f"%{event}%")

        athlete_pbs = pd.read_sql(query, conn, params=params)
        conn.close()

        if athlete_pbs.empty:
            return gaps

        # Load standards
        if not os.path.exists(self.wittw_db):
            return gaps

        conn = sqlite3.connect(self.wittw_db)
        try:
            standards_men = pd.read_sql("SELECT * FROM standards_men", conn)
        except:
            standards_men = pd.DataFrame()
        conn.close()

        # Analyze each event
        for _, row in athlete_pbs.iterrows():
            event_name = row['event_name']
            pb_str = row['pb_result']
            is_field = self._is_field_event(event_name)
            pb_val = self._parse_mark(pb_str, is_field)

            if pb_val is None:
                continue

            # Find matching standard
            matching = standards_men[standards_men['Event'].str.lower().str.contains(
                event_name.lower().replace(' ', '-').replace('m', '-metres'),
                na=False
            )]

            if matching.empty:
                # Try reverse match
                for _, std_row in standards_men.iterrows():
                    if std_row['Event'].lower().replace('-', ' ') in event_name.lower():
                        matching = pd.DataFrame([std_row])
                        break

            if matching.empty:
                continue

            std_row = matching.iloc[0]
            gold_val = std_row.get('Gold_Raw')
            if pd.isna(gold_val):
                continue

            # Calculate gap
            if is_field:
                gap = gold_val - pb_val  # Need to gain distance
            else:
                gap = pb_val - gold_val  # Need to drop time

            gap_pct = (abs(gap) / gold_val) * 100 if gold_val else 0

            # Determine training priority
            discipline = self._get_discipline_for_event(event_name)
            training_priority = self._get_training_priority(discipline, abs(gap), is_field)

            # Estimate improvement potential
            improvement_potential = self._estimate_improvement_potential(gap_pct)

            gaps.append(PerformanceGap(
                athlete_name=athlete_name,
                event=event_name,
                current_pb=pb_val,
                target_standard='gold',
                target_mark=gold_val,
                gap=gap,
                gap_percentage=gap_pct,
                training_priority=training_priority,
                estimated_improvement_potential=improvement_potential,
                is_field_event=is_field
            ))

        return gaps

    def _get_training_priority(self, discipline: str, gap: float, is_field: bool) -> str:
        """Get training priority based on gap size."""
        if discipline not in COACHING_INSIGHTS:
            return "General athletic development"

        priorities = COACHING_INSIGHTS[discipline].get('training_priorities', {})

        # Normalize gap for comparison
        if is_field:
            if gap > 5:
                key = next((k for k in priorities if '> 5' in k or '> 0.50' in k), None)
            elif gap > 2:
                key = next((k for k in priorities if '2-5' in k or '0.20-0.50' in k), None)
            else:
                key = next((k for k in priorities if '< 2' in k or '< 0.20' in k), None)
        else:
            if gap > 0.5:
                key = next((k for k in priorities if '> 0.5' in k or '> 5s' in k or '> 3min' in k), None)
            elif gap > 0.2:
                key = next((k for k in priorities if '0.3-0.5' in k or '0.2-0.5' in k or '2-5s' in k or '1-3min' in k), None)
            else:
                key = next((k for k in priorities if '< 0.1' in k or '< 0.2' in k or '< 2s' in k or '< 1min' in k), None)

        if key and key in priorities:
            return ', '.join(priorities[key])
        return "Individualized program required"

    def _estimate_improvement_potential(self, gap_pct: float) -> str:
        """Estimate realistic improvement potential."""
        if gap_pct > 10:
            return "Long-term development (3-5 years)"
        elif gap_pct > 5:
            return "Medium-term pathway (2-3 years)"
        elif gap_pct > 2:
            return "Short-term achievable (1-2 years)"
        else:
            return "Breakthrough imminent (< 1 year)"

    def get_round_progression_advice(self, event: str, current_pb: float) -> Dict:
        """Get advice for navigating championship rounds."""
        discipline = self._get_discipline_for_event(event)

        if discipline == 'sprints':
            progression = COACHING_INSIGHTS['sprints']['round_progression']
            event_key = event.replace('-metres', 'm').replace('metres', 'm')

            for key in progression:
                if key in event_key or event_key in key:
                    prog = progression[key]
                    return {
                        'event': event,
                        'current_pb': current_pb,
                        'heats_target': current_pb * prog['pct_save_heats'],
                        'heats_expected_improvement': prog['heats_to_semis'],
                        'semis_target': current_pb - prog['heats_to_semis'],
                        'final_target': current_pb - prog['heats_to_semis'] - prog['semis_to_final'],
                        'advice': f"Save energy in heats (run {prog['pct_save_heats']*100:.0f}% effort). "
                                  f"Expect {prog['heats_to_semis']:.2f}s improvement heats->semis, "
                                  f"{prog['semis_to_final']:.2f}s semis->final."
                    }

        return {'event': event, 'advice': 'Event-specific round progression data not available'}

    def get_tactical_recommendations(self, event: str, athlete_profile: Dict = None) -> Dict:
        """Get tactical recommendations for racing."""
        discipline = self._get_discipline_for_event(event)

        if discipline == 'middle_distance':
            tactics = COACHING_INSIGHTS['middle_distance']['tactical_profiles']
            splits = COACHING_INSIGHTS['middle_distance']['split_analysis']

            event_key = event.replace('-metres', 'm').replace('metres', 'm')
            split_advice = splits.get(event_key, splits.get('800m', {}))

            return {
                'event': event,
                'tactical_options': tactics,
                'split_guidance': split_advice,
                'recommendation': "Choose tactic based on athlete's strengths: "
                                  "front-runner for speed-endurance types, "
                                  "sit-and-kick for 400m-speed types."
            }

        return {'event': event, 'tactical_options': 'Standard race execution'}

    def generate_development_report(self, athlete_name: str) -> Dict:
        """Generate comprehensive development report for an athlete."""
        gaps = self.analyze_athlete_gap(athlete_name)

        if not gaps:
            return {'athlete': athlete_name, 'status': 'No performance data available'}

        report = {
            'athlete': athlete_name,
            'analysis_date': datetime.now().isoformat(),
            'events_analyzed': len(gaps),
            'performance_gaps': [],
            'priority_training_areas': [],
            'timeline_assessment': {}
        }

        for gap in gaps:
            gap_info = {
                'event': gap.event,
                'current_pb': self._format_mark(gap.current_pb, gap.event),
                'gold_standard': self._format_mark(gap.target_mark, gap.event),
                'gap': self._format_mark(abs(gap.gap), gap.event),
                'gap_percentage': f"{gap.gap_percentage:.1f}%",
                'training_focus': gap.training_priority,
                'timeline': gap.estimated_improvement_potential
            }
            report['performance_gaps'].append(gap_info)

            if gap.training_priority not in report['priority_training_areas']:
                report['priority_training_areas'].append(gap.training_priority)

        # Overall timeline
        min_gap_pct = min(g.gap_percentage for g in gaps)
        if min_gap_pct < 2:
            report['timeline_assessment'] = {
                'status': 'Medal Contender',
                'message': 'Within striking distance of global medals. Focus on championship peaking.',
                'priority': 'Championship preparation, marginal gains, mental skills'
            }
        elif min_gap_pct < 5:
            report['timeline_assessment'] = {
                'status': 'Emerging Talent',
                'message': 'On pathway to global competitiveness. Consistent development key.',
                'priority': 'Structured training blocks, competition exposure, technical refinement'
            }
        else:
            report['timeline_assessment'] = {
                'status': 'Development Phase',
                'message': 'Building foundation for future breakthroughs.',
                'priority': 'Base conditioning, technique establishment, competition experience'
            }

        return report

    def get_discipline_knowledge(self, discipline: str) -> Dict:
        """Access deep discipline knowledge for coaching decisions."""
        if discipline in COACHING_INSIGHTS:
            return COACHING_INSIGHTS[discipline]
        return {'error': f'Unknown discipline: {discipline}'}


def print_coaching_report(athlete_name: str):
    """Print formatted coaching report."""
    ci = CoachingIntelligence()
    report = ci.generate_development_report(athlete_name)

    print("=" * 70)
    print(f"COACHING INTELLIGENCE REPORT: {report['athlete']}")
    print(f"Generated: {report['analysis_date']}")
    print("=" * 70)

    if report.get('status') == 'No performance data available':
        print("No data available for this athlete.")
        return

    print(f"\nEvents Analyzed: {report['events_analyzed']}")
    print("\n--- PERFORMANCE GAPS ---")

    for gap in report['performance_gaps']:
        print(f"\n{gap['event']}:")
        print(f"  Current PB: {gap['current_pb']}")
        print(f"  Gold Standard: {gap['gold_standard']}")
        print(f"  Gap: {gap['gap']} ({gap['gap_percentage']})")
        print(f"  Training Focus: {gap['training_focus']}")
        print(f"  Timeline: {gap['timeline']}")

    print("\n--- PRIORITY TRAINING AREAS ---")
    for area in report['priority_training_areas']:
        print(f"  * {area}")

    print(f"\n--- OVERALL ASSESSMENT ---")
    timeline = report['timeline_assessment']
    print(f"Status: {timeline['status']}")
    print(f"Message: {timeline['message']}")
    print(f"Priority: {timeline['priority']}")
    print("=" * 70)


if __name__ == '__main__':
    # Example: Analyze a KSA athlete
    ci = CoachingIntelligence()

    # List available athletes
    if os.path.exists(ci.profiles_db):
        conn = sqlite3.connect(ci.profiles_db)
        athletes = pd.read_sql("SELECT DISTINCT full_name FROM ksa_athletes", conn)
        conn.close()

        print("KSA Athletes in Database:")
        for i, name in enumerate(athletes['full_name'].tolist()[:10]):
            print(f"  {i+1}. {name}")
            print_coaching_report(name)
            print("\n")
