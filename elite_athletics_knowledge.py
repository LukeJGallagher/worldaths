"""
Elite Athletics Knowledge Base
Comprehensive discipline-specific expertise for world-class athletics analysis

This module provides:
- Deep technical knowledge for each event
- Elite terminology and jargon
- Performance analysis frameworks
- Biomechanical insights
- Tactical considerations
- Common coaching cues and corrections
- World-class benchmarks and standards

Designed for use with AI chatbots and analytics systems.
"""

from typing import Dict, List, Any


###################################
# SPRINT EVENTS (100m, 200m, 400m)
###################################

SPRINTS_KNOWLEDGE = {
    "100m": {
        "overview": """The 100m dash is the premier speed event in athletics. It requires near-perfect
        reaction time, explosive acceleration, maximum velocity development, and speed maintenance.""",

        "phases": {
            "reaction_phase": {
                "description": "Time from gun to first movement",
                "elite_standard": "0.12-0.15 seconds",
                "false_start_threshold": "<0.100 seconds (impossible human reaction)",
                "key_factors": ["Anticipation", "Neural readiness", "Starting block setup"],
                "coaching_cues": ["Drive low", "Push don't pull", "Patience at the gun"]
            },
            "block_clearance": {
                "description": "First 2 strides out of blocks",
                "elite_standard": "First foot contact at 1.0-1.1m from line",
                "key_factors": ["Block angles (front 45-50°, rear 70-80°)", "Push duration", "Arm action"],
                "coaching_cues": ["Stay low", "Push the track away", "Aggressive arms"]
            },
            "drive_phase": {
                "description": "Acceleration from 0-30m",
                "elite_standard": "Reach 95% max velocity by 50-60m",
                "key_factors": ["Ground contact angle", "Hip extension power", "Gradually rising torso"],
                "coaching_cues": ["Push back, not down", "Patient rise", "Drive knees forward"]
            },
            "transition_phase": {
                "description": "From drive to upright (30-50m)",
                "key_factors": ["Smooth posture change", "Maintain acceleration", "Relaxation begins"],
                "coaching_cues": ["Run tall", "Float the arms", "Relax face and jaw"]
            },
            "maximum_velocity": {
                "description": "Peak speed phase (50-80m)",
                "elite_standard": "11.5-12.5+ m/s for elite men, 10.5-11.5+ m/s for elite women",
                "key_factors": ["Stride length (2.40-2.60m elite men)", "Stride frequency (4.5-5.0 Hz)",
                               "Ground contact time (<0.085s)", "Flight time"],
                "coaching_cues": ["Tall and relaxed", "Quick feet", "Run through the finish"]
            },
            "speed_maintenance": {
                "description": "Minimizing deceleration (80-100m)",
                "key_factors": ["Relaxation technique", "Metabolic efficiency", "Fatigue resistance"],
                "common_problems": ["Early deceleration", "Tightening up", "Over-striding"],
                "coaching_cues": ["Stay loose", "Maintain frequency", "Pump don't push"]
            }
        },

        "biomechanics": {
            "ground_contact_time": "Elite: <0.085 seconds at max velocity",
            "flight_time": "Elite: ~0.12 seconds",
            "stride_length_range": "Elite men: 2.40-2.65m at max velocity",
            "stride_frequency": "Elite: 4.5-5.2 Hz",
            "horizontal_force_production": "Critical for acceleration",
            "vertical_oscillation": "Minimize - should be <5cm at max velocity"
        },

        "benchmarks": {
            "men": {
                "world_record": 9.58,
                "world_class": 9.95,
                "national_class": 10.20,
                "regional_elite": 10.50,
                "good_club": 11.00,
                "reaction_time": {"excellent": 0.13, "good": 0.15, "average": 0.17}
            },
            "women": {
                "world_record": 10.49,
                "world_class": 10.90,
                "national_class": 11.30,
                "regional_elite": 11.70,
                "good_club": 12.20,
                "reaction_time": {"excellent": 0.14, "good": 0.16, "average": 0.18}
            }
        },

        "championship_rounds": {
            "heats": "Top 3 + fastest losers advance (usually 8 heats to 24 athletes)",
            "semi_finals": "Top 2 + 2 fastest losers advance (3 semis to 8 athletes)",
            "final": "8 athletes, lane draw based on semis performance",
            "typical_improvement": {"heat_to_semi": 0.05, "semi_to_final": 0.08}
        },

        "external_factors": {
            "wind": {
                "legal_limit": 2.0,
                "wind_benefit": "~0.05s per 1.0 m/s tailwind",
                "headwind_cost": "~0.06s per 1.0 m/s headwind"
            },
            "altitude": {
                "benefit": "~0.03s improvement above 1000m due to reduced air resistance",
                "notable_venues": "Mexico City (2250m), Nairobi (1795m), Johannesburg (1753m)"
            },
            "temperature": {
                "optimal": "20-25°C",
                "cold_impact": "Reduced muscle elasticity, slower times",
                "heat_impact": "Warm muscles beneficial, but dehydration risk"
            },
            "track_surface": "Different brands (Mondotrack, Regupol, etc.) have varying stiffness/rebound"
        },

        "terminology": {
            "FAT": "Fully Automatic Timing - electronic timing to 0.01s accuracy",
            "RT": "Reaction Time - time from gun to leaving blocks",
            "w": "Wind reading in m/s (negative = headwind)",
            "DNS": "Did Not Start",
            "DNF": "Did Not Finish",
            "DQ": "Disqualified (usually false start)",
            "PB": "Personal Best",
            "SB": "Season Best",
            "WL": "World Lead (best mark in current year)",
            "AR": "Area Record (continental record)",
            "NR": "National Record",
            "WR": "World Record",
            "OR": "Olympic Record",
            "CR": "Championship Record"
        }
    },

    "200m": {
        "overview": """The 200m combines the speed of the 100m with the curve running skill and
        speed endurance. The staggered start and curve running make lane draw significant.""",

        "phases": {
            "blocks_and_curve": {
                "description": "Start through first 100m on curve",
                "key_factors": ["Block placement on curve", "Lean into curve", "Relaxed arms on curve"],
                "coaching_cues": ["Run the bend with your body", "Don't fight the curve", "Stay low longer"]
            },
            "curve_to_straight_transition": {
                "description": "Coming off the curve (100-110m)",
                "key_factors": ["Smooth transition", "Maintain speed", "Don't drift wide"],
                "coaching_cues": ["Run off the curve", "Hold your line", "Gradual straightening"]
            },
            "home_straight": {
                "description": "Final 90m to finish",
                "key_factors": ["Speed maintenance", "Relaxation", "Running form under fatigue"],
                "coaching_cues": ["Run tall", "Stay relaxed", "Quick arms"]
            }
        },

        "curve_running": {
            "left_arm_action": "Shorter, more across body",
            "right_arm_action": "Longer, more forward",
            "body_lean": "5-8° into the curve",
            "foot_strike": "Push from inside of left foot, outside of right",
            "lane_advantage": "Inner lanes: tighter curve, less distance to accelerate; Outer lanes: gentler curve but can't see competitors"
        },

        "benchmarks": {
            "men": {
                "world_record": 19.19,
                "world_class": 20.00,
                "national_class": 20.50,
                "regional_elite": 21.20,
                "100m_conversion": "Double 100m time + 0.95s (approx)"
            },
            "women": {
                "world_record": 21.34,
                "world_class": 22.30,
                "national_class": 23.00,
                "regional_elite": 23.80,
                "100m_conversion": "Double 100m time + 1.05s (approx)"
            }
        },

        "split_analysis": {
            "elite_curve_time": "10.5-10.9s (first 100m for elite men)",
            "home_straight_time": "9.3-9.8s (second 100m for elite men)",
            "ideal_negative_split": "First half faster by 0.8-1.0s due to curve"
        }
    },

    "400m": {
        "overview": """The 400m is the longest sprint, requiring optimal distribution of anaerobic
        energy systems. It's often called the hardest race in athletics due to the extreme lactate
        accumulation and speed endurance demands.""",

        "energy_systems": {
            "ATP-PC": "First 6-8 seconds (approximately first 60m)",
            "glycolytic": "Dominant from 8-45 seconds (60-350m)",
            "aerobic_contribution": "~30-35% of total energy",
            "lactate_peak": "Typically 10-12 seconds post-race, 20+ mmol/L"
        },

        "phases": {
            "first_curve": {
                "distance": "0-100m",
                "strategy": "Controlled aggression, don't over-accelerate",
                "elite_split": "10.8-11.2s for elite men"
            },
            "back_straight": {
                "distance": "100-200m",
                "strategy": "Float and relax, maintain rhythm",
                "elite_split": "10.2-10.6s for elite men"
            },
            "second_curve": {
                "distance": "200-300m",
                "strategy": "Most critical phase - resist deceleration",
                "elite_split": "10.8-11.4s for elite men",
                "coaching_cues": ["Stay tall", "Drive the curve", "Maintain arm action"]
            },
            "home_straight": {
                "distance": "300-400m",
                "strategy": "Speed maintenance under extreme fatigue",
                "elite_split": "11.5-12.8s for elite men",
                "common_issues": ["Tie up", "Stride shortening", "Form breakdown"]
            }
        },

        "split_strategies": {
            "even_pace": "Each 100m similar pace - safest but not fastest",
            "negative_split": "Second half faster - rare in 400m",
            "positive_split": "First half faster - most common for elites",
            "optimal_ratio": "First 200m to second 200m = 1.0 to 1.03-1.06"
        },

        "benchmarks": {
            "men": {
                "world_record": 43.03,
                "world_class": 44.50,
                "national_class": 46.00,
                "regional_elite": 47.50,
                "first_200m_elite": "21.0-21.5s",
                "speed_reserve": "Must have 20.5 or better 200m to run sub-44"
            },
            "women": {
                "world_record": 47.60,
                "world_class": 50.00,
                "national_class": 52.00,
                "regional_elite": 54.00,
                "first_200m_elite": "23.5-24.0s"
            }
        },

        "terminology": {
            "rigor": "The feeling of legs locking up in the final 100m due to lactate",
            "tie_up": "When muscles stop responding effectively in final straight",
            "float": "Relaxed running phase, typically on back straight",
            "bear_down": "Push through fatigue in final phase"
        }
    }
}


###################################
# MIDDLE DISTANCE (800m, 1500m, Mile)
###################################

MIDDLE_DISTANCE_KNOWLEDGE = {
    "800m": {
        "overview": """The 800m is a unique blend of speed and endurance, requiring both anaerobic
        power (55-60% contribution) and aerobic capacity (40-45%). Tactical awareness and finishing
        speed are critical at the elite level.""",

        "energy_systems": {
            "anaerobic_contribution": "55-60%",
            "aerobic_contribution": "40-45%",
            "lactate_levels": "Peak 15-20 mmol/L post-race",
            "vo2_requirement": "Elite: 70+ ml/kg/min"
        },

        "race_tactics": {
            "front_running": {
                "description": "Lead from the start, set honest pace",
                "pros": ["Control race tempo", "Avoid traffic", "Impose your rhythm"],
                "cons": ["Wind resistance", "Target for kickers", "Must judge pace perfectly"],
                "famous_example": "David Rudisha's 2012 Olympic WR"
            },
            "sit_and_kick": {
                "description": "Stay in contact, unleash finishing speed",
                "pros": ["Draft benefit", "Watch competitors", "Save energy for kick"],
                "cons": ["Risk of being boxed", "Slow pace doesn't suit all", "Must have superior kick"],
                "famous_example": "Nijel Amos's typical tactics"
            },
            "mid_pack_tactics": {
                "description": "Stay in striking position throughout",
                "pros": ["Flexibility to respond", "Avoid box-in situations"],
                "cons": ["Compromise strategy", "React rather than dictate"]
            }
        },

        "split_analysis": {
            "world_record_men": {"first_400": 49.28, "second_400": 51.35, "total": "1:40.91"},
            "typical_elite_men": {"first_400": "50-51s", "second_400": "52-54s"},
            "tactical_race_split": "55-56s first lap, 52-54s second lap (fast finish)",
            "fast_race_split": "49-50s first lap, 51-53s second lap"
        },

        "lane_draw": {
            "waterfall_start": "After 100m (one curve), athletes break for inside",
            "strategy": "Need to establish position before first lap complete",
            "risks": ["Getting boxed", "Wide running", "Too aggressive position fight"]
        },

        "benchmarks": {
            "men": {
                "world_record": "1:40.91",
                "world_class": "1:44.00",
                "national_class": "1:47.00",
                "regional_elite": "1:50.00",
                "400m_requirement": "Under 47s ideal for sub-1:44"
            },
            "women": {
                "world_record": "1:53.28",
                "world_class": "1:57.00",
                "national_class": "2:02.00",
                "regional_elite": "2:06.00"
            }
        },

        "terminology": {
            "kick": "Final acceleration, typically last 150-200m",
            "break": "Waterfall break point where athletes move to inside lane",
            "honest_pace": "Fast, evenly-paced race (not tactical)",
            "rabbited_race": "Using a pacemaker to ensure fast time"
        }
    },

    "1500m": {
        "overview": """The 1500m (metric mile) is the classic middle distance event requiring
        tactical intelligence, aerobic capacity, and a devastating finishing kick. Often
        called the thinking person's race.""",

        "energy_systems": {
            "anaerobic_contribution": "20-25%",
            "aerobic_contribution": "75-80%",
            "vo2_requirement": "Elite: 75+ ml/kg/min",
            "running_economy": "Critical factor at elite level"
        },

        "race_phases": {
            "first_400m": {
                "typical_elite": "54-58s",
                "strategy": "Establish position, settle into rhythm",
                "common_mistake": "Too fast start, wastes energy"
            },
            "middle_800m": {
                "typical_elite": "2:00-2:04 for this section",
                "strategy": "Maintain contact with leaders, conserve energy",
                "coaching_cue": "Run relaxed, stay patient"
            },
            "bell_lap": {
                "description": "Final 400m announced by bell",
                "typical_elite": "52-56s for final lap",
                "tactics": "Position for final kick, move up if needed"
            },
            "final_200m": {
                "description": "The kick",
                "elite_standard": "26-28s for final 200m",
                "key_factor": "Speed reserve - must have 400m speed to kick"
            }
        },

        "pacemaking": {
            "diamond_league_standard": "Usually 2-3 pacemakers, drop at 1000-1100m",
            "typical_pace": "Through 800m in ~1:52-1:55 for sub-3:30 attempt",
            "1000m_split": "~2:22-2:25 for sub-3:30",
            "without_pacemaker": "Tactical race, usually slower overall"
        },

        "benchmarks": {
            "men": {
                "world_record": "3:26.00",
                "world_class": "3:32.00",
                "national_class": "3:40.00",
                "regional_elite": "3:48.00",
                "800m_requirement": "Sub-1:46 typical for sub-3:32"
            },
            "women": {
                "world_record": "3:49.11",
                "world_class": "4:00.00",
                "national_class": "4:10.00",
                "regional_elite": "4:20.00"
            }
        },

        "mile_conversion": {
            "1500m_to_mile": "Add 17-18 seconds for equivalent mile",
            "example": "3:30 1500m ≈ 3:47-3:48 mile"
        }
    }
}


###################################
# LONG DISTANCE (5000m, 10000m, Marathon)
###################################

LONG_DISTANCE_KNOWLEDGE = {
    "5000m": {
        "overview": """The 5000m (12.5 laps) demands high aerobic capacity, excellent running
        economy, and the ability to respond to surges. Modern championship races feature
        tactical changes of pace.""",

        "energy_systems": {
            "aerobic_contribution": "90-95%",
            "anaerobic_contribution": "5-10% (used in surges and finish)",
            "vo2_max_requirement": "Elite: 80+ ml/kg/min",
            "lactate_threshold": "Must be very close to race pace"
        },

        "pacing_strategies": {
            "even_pace": {
                "description": "Each kilometer at same pace",
                "example": "12:50 = 5 × 2:34 per km",
                "use_case": "Time trials, record attempts"
            },
            "negative_split": {
                "description": "Second half faster than first",
                "difficulty": "Very hard to execute in championship races"
            },
            "tactical_racing": {
                "description": "Respond to surges, wait for kick",
                "championship_style": "Common in Olympics, World Championships",
                "typical_pattern": "Slow early, fast final 2-3 laps"
            }
        },

        "surges": {
            "purpose": "Break competitors, test fitness",
            "typical_timing": ["With 5-6 laps to go", "With 3 laps to go", "Bell lap"],
            "execution": "5-10 second burst at 800m pace, then settle"
        },

        "benchmarks": {
            "men": {
                "world_record": "12:35.36",
                "world_class": "13:00.00",
                "national_class": "13:30.00",
                "regional_elite": "14:00.00",
                "per_lap_wr_pace": "~60.5 seconds per 400m"
            },
            "women": {
                "world_record": "14:00.21",
                "world_class": "14:45.00",
                "national_class": "15:30.00",
                "regional_elite": "16:15.00"
            }
        }
    },

    "10000m": {
        "overview": """The 10000m (25 laps) is the ultimate track endurance event, requiring
        exceptional aerobic development, mental toughness, and tactical awareness over
        25+ minutes of racing.""",

        "race_dynamics": {
            "typical_pattern": "Large pack through 6-7km, surges thin the field",
            "key_moments": ["5km split", "With 8-10 laps to go", "Final 2km"],
            "lapping": "In championship races, leaders may lap back-markers"
        },

        "split_targets": {
            "sub_27_men": {
                "per_km": "2:42",
                "per_400m": "64.8s",
                "5km_split": "13:30"
            },
            "sub_30_women": {
                "per_km": "3:00",
                "per_400m": "72s",
                "5km_split": "15:00"
            }
        },

        "benchmarks": {
            "men": {
                "world_record": "26:11.00",
                "world_class": "27:00.00",
                "national_class": "28:00.00",
                "regional_elite": "29:30.00"
            },
            "women": {
                "world_record": "28:54.14",
                "world_class": "30:30.00",
                "national_class": "32:00.00",
                "regional_elite": "34:00.00"
            }
        }
    },

    "marathon": {
        "overview": """The marathon (42.195km / 26.2 miles) is the ultimate endurance test,
        requiring years of aerobic base building, optimal pacing strategy, fueling, and
        mental fortitude.""",

        "energy_demands": {
            "calories_burned": "2500-3000+ calories",
            "glycogen_stores": "Depleted around 30-35km without fueling",
            "fat_oxidation": "Critical for fuel efficiency"
        },

        "the_wall": {
            "typical_location": "30-35km mark",
            "cause": "Glycogen depletion",
            "prevention": "Proper pacing, in-race fueling, training adaptation"
        },

        "pacing_strategies": {
            "negative_split": {
                "description": "Second half faster",
                "difficulty": "Rare but effective when executed",
                "example": "Kipchoge often runs this way"
            },
            "even_splits": {
                "description": "Same pace throughout",
                "target": "World class men: 2:55/km, women: 3:15/km"
            },
            "banking_time": {
                "warning": "Going out too fast almost never works",
                "typical_result": "Dramatic slowdown after 30km"
            }
        },

        "race_day_factors": {
            "temperature": "Optimal 5-12°C, performance drops significantly above 15°C",
            "humidity": "Below 60% ideal",
            "course_profile": "Net downhill courses faster but harder on quads",
            "pacers": "Super shoes and formations now common in fast races"
        },

        "benchmarks": {
            "men": {
                "world_record": "2:00:35",
                "world_class": "2:06:00",
                "national_class": "2:12:00",
                "boston_qualifying": "Age-dependent, 3:00-3:05 for most men"
            },
            "women": {
                "world_record": "2:11:53",
                "world_class": "2:20:00",
                "national_class": "2:30:00",
                "boston_qualifying": "Age-dependent, 3:30-3:35 for most women"
            }
        },

        "terminology": {
            "bonk": "Sudden energy depletion (hitting the wall)",
            "BQ": "Boston Qualifying time",
            "negative_split": "Second half faster than first",
            "rabbit": "Pacemaker",
            "tangent": "Shortest line on the course",
            "aid_station": "Water/fuel points on course"
        }
    }
}


###################################
# HURDLES (100mH, 110mH, 400mH)
###################################

HURDLES_KNOWLEDGE = {
    "110mH_men": {
        "overview": """The 110m hurdles requires explosive sprinting speed combined with
        exceptional technique over 10 hurdles. Lead leg speed, trail leg mechanics, and
        rhythm maintenance are critical.""",

        "technical_specs": {
            "hurdle_height": "1.067m (3' 6\")",
            "first_hurdle": "13.72m from start",
            "hurdle_spacing": "9.14m between hurdles",
            "last_hurdle_to_finish": "14.02m",
            "number_of_hurdles": 10
        },

        "stride_pattern": {
            "to_first_hurdle": "8 strides (7 steps between block contact and takeoff)",
            "between_hurdles": "3 strides",
            "elite_ground_contact": "<0.11 seconds on touchdown"
        },

        "technique_components": {
            "lead_leg": {
                "action": "Snap forward and down",
                "key_cue": "Toe up, knee up, snap down",
                "common_error": "Wrapping leg around hurdle"
            },
            "trail_leg": {
                "action": "Flat, knee to armpit, pull through",
                "key_cue": "Knee up and through, not around",
                "common_error": "Dropping trail leg knee"
            },
            "arm_action": {
                "lead_arm": "Punch forward for balance",
                "trail_arm": "Back and then cycle forward",
                "common_error": "Arms crossing midline"
            },
            "body_lean": "5-10° forward lean at takeoff"
        },

        "benchmarks": {
            "world_record": 12.80,
            "world_class": 13.20,
            "national_class": 13.60,
            "regional_elite": 14.20,
            "100m_correlation": "Must be sub-10.5 flat to run sub-13.00 hurdles"
        }
    },

    "100mH_women": {
        "overview": """The women's 100m hurdles features lower hurdles than the men's 110m
        but with similar technical demands. Speed between hurdles is crucial.""",

        "technical_specs": {
            "hurdle_height": "0.838m (2' 9\")",
            "first_hurdle": "13.00m from start",
            "hurdle_spacing": "8.50m between hurdles",
            "last_hurdle_to_finish": "10.50m",
            "number_of_hurdles": 10
        },

        "stride_pattern": {
            "to_first_hurdle": "8 strides typically",
            "between_hurdles": "3 strides",
            "adjustment": "Some athletes use 7 to first hurdle for faster start"
        },

        "benchmarks": {
            "world_record": 12.12,
            "world_class": 12.60,
            "national_class": 13.20,
            "regional_elite": 13.80
        }
    },

    "400mH": {
        "overview": """The 400m hurdles combines 400m speed endurance with hurdling technique
        over 10 hurdles. Stride pattern adjustment under fatigue and curve hurdling add
        complexity.""",

        "technical_specs": {
            "hurdle_height_men": "0.914m (3' 0\")",
            "hurdle_height_women": "0.762m (2' 6\")",
            "first_hurdle": "45m from start",
            "hurdle_spacing": "35m between hurdles",
            "last_hurdle_to_finish": "40m",
            "number_of_hurdles": 10
        },

        "stride_patterns": {
            "elite_men": {
                "to_first": "21-22 strides",
                "between": "13 strides (elite), 14 (developing), 15 (alternate leg)"
            },
            "elite_women": {
                "to_first": "23-24 strides",
                "between": "15 strides (elite), 16-17 (developing)"
            },
            "fatigue_adjustment": "Most athletes add 1-2 strides per hurdle interval in second half"
        },

        "race_phases": {
            "first_5_hurdles": "Establish rhythm, controlled aggression",
            "hurdles_6_8": "Critical - maintain stride pattern or adjust smoothly",
            "hurdles_9_10": "Fight fatigue, maintain technique",
            "run_in": "40m to finish - longest run-in in hurdle events"
        },

        "benchmarks": {
            "men": {
                "world_record": 45.94,
                "world_class": 47.50,
                "national_class": 49.50,
                "400m_flat_requirement": "Must be sub-46 flat for sub-48 hurdles"
            },
            "women": {
                "world_record": 50.68,
                "world_class": 53.00,
                "national_class": 56.00
            }
        }
    }
}


###################################
# JUMPS (HJ, LJ, TJ, PV)
###################################

JUMPS_KNOWLEDGE = {
    "high_jump": {
        "overview": """High jump requires approach speed, explosive takeoff power, and
        exceptional technique to clear the bar using the Fosbury Flop. The J-curve
        approach and one-foot takeoff are defining characteristics.""",

        "approach": {
            "distance": "8-12 strides typical",
            "pattern": "J-curve approach from side",
            "speed": "Build to ~7-8 m/s at takeoff",
            "penultimate_step": "Longer, flatter, sets up takeoff"
        },

        "takeoff": {
            "foot_position": "Heel-ball-toe rock on takeoff foot",
            "arm_action": "Synchronized drive upward",
            "conversion": "Horizontal to vertical momentum",
            "ground_contact": "0.14-0.18 seconds"
        },

        "bar_clearance": {
            "arch": "Maximum arch at bar peak",
            "head_action": "Look back, tuck chin",
            "hip_lift": "Critical for clearance height",
            "leg_kick": "Snap legs up after hips clear"
        },

        "competition_format": {
            "attempts": "3 attempts per height",
            "progression": "Bar raised after each round",
            "countback": "Ties broken by fewer misses at last cleared height, then overall misses",
            "passing": "Tactical passes can provide advantage"
        },

        "benchmarks": {
            "men": {
                "world_record": 2.45,
                "world_class": 2.30,
                "national_class": 2.20,
                "regional_elite": 2.10
            },
            "women": {
                "world_record": 2.09,
                "world_class": 1.95,
                "national_class": 1.85,
                "regional_elite": 1.75
            }
        }
    },

    "long_jump": {
        "overview": """Long jump combines maximum speed with precise takeoff mechanics.
        The approach, board contact, flight, and landing must be perfectly coordinated
        for optimal distance.""",

        "phases": {
            "approach": {
                "distance": "18-22 strides (40-50m)",
                "speed": "Elite men reach 10.5-11+ m/s",
                "checkmarks": "Usually 2 (start and mid-point)",
                "acceleration_pattern": "Build throughout approach"
            },
            "takeoff": {
                "board": "20cm wide, foul line at front edge",
                "foot_contact": "Flat-foot, slightly ahead of CM",
                "takeoff_angle": "18-22° optimal",
                "contact_time": "0.10-0.12 seconds",
                "free_leg_drive": "Knee punch upward"
            },
            "flight": {
                "techniques": ["Sail/Hang", "Hitch-kick (2.5, 3.5 strides)", "Hang"],
                "purpose": "Maintain balance, prepare for landing",
                "arm_action": "Circular for balance"
            },
            "landing": {
                "leg_position": "Extend forward",
                "arm_action": "Swing forward for momentum",
                "hip_position": "Drive past feet on contact"
            }
        },

        "wind_rules": {
            "legal_limit": 2.0,
            "adjustment": "~0.10m per 1.0 m/s for elite jumps",
            "records": "Only valid with wind ≤ 2.0 m/s"
        },

        "benchmarks": {
            "men": {
                "world_record": 8.95,
                "world_class": 8.30,
                "national_class": 7.80,
                "regional_elite": 7.40,
                "100m_correlation": "Sub-10.3 typically needed for 8m+ jump"
            },
            "women": {
                "world_record": 7.52,
                "world_class": 6.90,
                "national_class": 6.40,
                "regional_elite": 6.00
            }
        }
    },

    "triple_jump": {
        "overview": """Triple jump requires hop-step-jump coordination from a single-leg
        takeoff. The phase balance, ground contacts, and posture maintenance are
        critical for elite performance.""",

        "phases": {
            "hop": {
                "description": "Takeoff and land on same foot",
                "phase_ratio": "35-37% of total distance",
                "key_focus": "Active landing, maintain speed",
                "ground_contact": "~0.13-0.14 seconds"
            },
            "step": {
                "description": "Land on opposite foot",
                "phase_ratio": "28-30% of total distance",
                "key_focus": "Balance, forward drive",
                "common_error": "Collapsing on landing"
            },
            "jump": {
                "description": "Final phase to pit",
                "phase_ratio": "33-37% of total distance",
                "key_focus": "Maximum effort, flight technique"
            }
        },

        "phase_ratios": {
            "balanced": "35% - 30% - 35%",
            "hop_dominant": "37% - 29% - 34% (power jumpers)",
            "jump_dominant": "34% - 29% - 37% (speed jumpers)"
        },

        "board_distances": {
            "men": "Usually 13m from pit",
            "women": "Usually 11m from pit",
            "elite_adjustment": "May move back for longer jumps"
        },

        "benchmarks": {
            "men": {
                "world_record": 18.29,
                "world_class": 17.20,
                "national_class": 16.00,
                "regional_elite": 15.00
            },
            "women": {
                "world_record": 15.74,
                "world_class": 14.50,
                "national_class": 13.50,
                "regional_elite": 12.50
            }
        }
    },

    "pole_vault": {
        "overview": """Pole vault is the most technically complex field event, requiring
        sprinting speed, gymnastics ability, and the skill to use a flexible pole as
        an energy storage and release system.""",

        "technical_phases": {
            "approach": {
                "distance": "16-20 steps (35-45m)",
                "speed": "Elite men: 9.5-10+ m/s at takeoff",
                "pole_carry": "Lower as approach progresses"
            },
            "plant": {
                "description": "Pole tip into box, hands drive up",
                "timing": "2-3 steps before takeoff",
                "common_error": "Early or late plant"
            },
            "takeoff": {
                "position": "Under top hand",
                "action": "Drive up, not out",
                "free_leg": "Continuous swing upward"
            },
            "swing_and_rockback": {
                "description": "Invert body on pole",
                "timing": "As pole begins to unbend",
                "body_position": "L-shape inversion"
            },
            "extension_and_turn": {
                "description": "Push off pole, rotate over bar",
                "timing": "At pole's maximum recoil",
                "technique": "Push-pull-turn sequence"
            },
            "bar_clearance": {
                "description": "Arch over bar, snap legs",
                "common_error": "Not waiting for full pole recoil"
            }
        },

        "pole_selection": {
            "factors": ["Weight rating", "Length", "Flex number"],
            "elite_men": "5.00-5.20m poles, rated for body weight + grip height",
            "grip_height": "Higher grip = more potential height, but harder to invert"
        },

        "benchmarks": {
            "men": {
                "world_record": 6.24,
                "world_class": 5.80,
                "national_class": 5.40,
                "regional_elite": 5.00
            },
            "women": {
                "world_record": 5.06,
                "world_class": 4.70,
                "national_class": 4.30,
                "regional_elite": 3.90
            }
        }
    }
}


###################################
# THROWS (SP, DT, HT, JT)
###################################

THROWS_KNOWLEDGE = {
    "shot_put": {
        "overview": """Shot put requires explosive power to accelerate a heavy implement
        (7.26kg men / 4kg women) from a stationary release to maximum velocity.
        Two techniques dominate: glide and rotational.""",

        "techniques": {
            "glide": {
                "description": "Linear movement across circle",
                "advantages": ["Easier to learn", "More consistent", "Better for tall athletes"],
                "elite_users": "Ryan Crouser (partially), historically Adam Nelson"
            },
            "rotational": {
                "description": "1.5-2 rotations across circle",
                "advantages": ["Higher velocity potential", "Uses whole body rotation"],
                "challenges": ["Balance", "Timing", "Fouling"],
                "elite_users": "Joe Kovacs, Tom Walsh"
            }
        },

        "circle_specs": {
            "diameter": "2.135m (7 feet)",
            "toe_board": "10cm high at front",
            "sector": "34.92° angle"
        },

        "release_mechanics": {
            "optimal_angle": "37-42° (lower than theoretical 45° due to release height)",
            "release_height": "Elite: 2.0-2.3m above ground",
            "release_velocity": "Elite men: 13.5-14.5 m/s"
        },

        "implement_specs": {
            "men": {"weight": "7.26kg", "diameter": "110-130mm"},
            "women": {"weight": "4.00kg", "diameter": "95-110mm"}
        },

        "benchmarks": {
            "men": {
                "world_record": 23.56,
                "world_class": 21.50,
                "national_class": 19.00,
                "regional_elite": 17.00
            },
            "women": {
                "world_record": 22.63,
                "world_class": 19.50,
                "national_class": 17.00,
                "regional_elite": 15.00
            }
        }
    },

    "discus": {
        "overview": """Discus throw requires rotational speed, timing, and aerodynamic
        understanding. The implement's flight characteristics make technique and wind
        reading crucial for optimal distance.""",

        "technique": {
            "starting_position": "Back of circle, facing away from sector",
            "winds": "1.5 rotations across circle",
            "delivery": "Low-to-high trajectory, wide sweep",
            "follow_through": "Rotation continues after release"
        },

        "aerodynamics": {
            "release_angle": "35-40° for men, 30-35° for women (lower due to lighter disc)",
            "angle_of_attack": "Slight nose-up into wind",
            "spin_rate": "7-10 revolutions per second",
            "wind_benefit": "Headwind can help distance (opposite of other throws)"
        },

        "implement_specs": {
            "men": {"weight": "2.0kg", "diameter": "219-221mm"},
            "women": {"weight": "1.0kg", "diameter": "180-182mm"}
        },

        "benchmarks": {
            "men": {
                "world_record": 74.08,
                "world_class": 68.00,
                "national_class": 60.00,
                "regional_elite": 52.00
            },
            "women": {
                "world_record": 76.80,
                "world_class": 67.00,
                "national_class": 58.00,
                "regional_elite": 50.00
            }
        }
    },

    "hammer": {
        "overview": """Hammer throw uses 3-4 rotational turns to accelerate the hammer
        (7.26kg men / 4kg women on 1.21m/1.19m wire) to maximum release velocity.
        Balance and acceleration through turns are critical.""",

        "technique": {
            "entry_swings": "2-3 preliminary swings to build momentum",
            "turns": "3-4 full rotations across circle",
            "acceleration": "Speed increases each turn",
            "release": "Low point release, not at high point"
        },

        "biomechanics": {
            "low_point": "Hammer passes lowest at right side (right-handed)",
            "high_point": "Hammer passes highest at left side",
            "radius": "Increasing through turns for more speed",
            "release_angle": "40-44°"
        },

        "implement_specs": {
            "men": {"weight": "7.26kg", "wire": "1.175-1.215m", "handle_diameter": "110-130mm"},
            "women": {"weight": "4.00kg", "wire": "1.160-1.195m", "handle_diameter": "110-130mm"}
        },

        "benchmarks": {
            "men": {
                "world_record": 86.74,
                "world_class": 78.00,
                "national_class": 70.00,
                "regional_elite": 60.00
            },
            "women": {
                "world_record": 82.98,
                "world_class": 73.00,
                "national_class": 65.00,
                "regional_elite": 55.00
            }
        }
    },

    "javelin": {
        "overview": """Javelin is the only throw with a run-up, combining sprinting speed
        with throwing power. The implement's aerodynamic design means technique and
        release angle are critical for optimal flight.""",

        "approach": {
            "distance": "30-35m for elite",
            "crossover_steps": "5-step pattern at end",
            "speed": "Build to ~7-8 m/s at release",
            "carry": "Above shoulder, palm up"
        },

        "delivery": {
            "withdrawal": "Draw javelin back during crossovers",
            "plant": "Block with front leg",
            "rotation": "Hip-torso-shoulder-arm sequence",
            "release": "Above and behind head"
        },

        "aerodynamics": {
            "release_angle": "33-36° optimal",
            "angle_of_attack": "Slight nose-up",
            "spin": "Should not wobble in flight",
            "landing": "Tip must strike ground first"
        },

        "implement_specs": {
            "men": {"weight": "800g", "length": "2.60-2.70m"},
            "women": {"weight": "600g", "length": "2.20-2.30m"}
        },

        "benchmarks": {
            "men": {
                "world_record": 98.48,
                "world_class": 85.00,
                "national_class": 75.00,
                "regional_elite": 65.00
            },
            "women": {
                "world_record": 72.28,
                "world_class": 64.00,
                "national_class": 55.00,
                "regional_elite": 48.00
            }
        }
    }
}


###################################
# COMBINED EVENTS
###################################

COMBINED_EVENTS_KNOWLEDGE = {
    "decathlon": {
        "overview": """The decathlon is the ultimate test of athletic versatility,
        comprising 10 events over two days. Points are scored using IAAF tables that
        balance different event types.""",

        "events": {
            "day_1": ["100m", "Long Jump", "Shot Put", "High Jump", "400m"],
            "day_2": ["110m Hurdles", "Discus", "Pole Vault", "Javelin", "1500m"]
        },

        "scoring": {
            "system": "IAAF Combined Events Scoring Tables",
            "world_record": "9126 points (Kevin Mayer)",
            "9000_point_standard": "Elite world class",
            "8000_point_standard": "National class"
        },

        "strategy": {
            "strength_events": "Maximize points in best events",
            "weakness_events": "Minimize damage, avoid zeros",
            "energy_management": "Balance effort across two days",
            "common_pitfalls": ["Fouling all attempts", "Going out too hard in 400m"]
        },

        "event_correlations": {
            "speed_events": "100m, Long Jump, 110mH correlate",
            "power_events": "Shot Put, Discus correlate",
            "technical_events": "Pole Vault, High Jump need technique focus",
            "endurance": "1500m requires specific training"
        },

        "recovery": {
            "day_1_to_day_2": "Nutrition, sleep, light stretching",
            "between_events": "Stay warm, hydrated, mentally focused"
        }
    },

    "heptathlon": {
        "overview": """The women's heptathlon is the female equivalent of the decathlon,
        featuring 7 events over two days. Speed and jumping events tend to dominate
        scoring.""",

        "events": {
            "day_1": ["100m Hurdles", "High Jump", "Shot Put", "200m"],
            "day_2": ["Long Jump", "Javelin", "800m"]
        },

        "scoring": {
            "world_record": "7291 points (Jackie Joyner-Kersee)",
            "7000_point_standard": "Elite world class",
            "6000_point_standard": "National class"
        },

        "key_factors": {
            "speed_based": "100mH, 200m, Long Jump often biggest point earners",
            "technical": "High Jump technique critical",
            "throws": "Shot Put and Javelin often weakest for speed-based athletes",
            "endurance": "800m is brutal after 6 events"
        }
    }
}


###################################
# GENERAL ATHLETICS TERMINOLOGY
###################################

ATHLETICS_TERMINOLOGY = {
    "timing_and_records": {
        "FAT": "Fully Automatic Timing - electronic to 0.01s",
        "HT": "Hand Timing - add 0.24s to convert to FAT equivalent",
        "PB": "Personal Best",
        "SB": "Season Best",
        "WR": "World Record",
        "OR": "Olympic Record",
        "CR": "Championship Record",
        "AR": "Area/Continental Record",
        "NR": "National Record",
        "WL": "World Lead (best of current year)"
    },

    "race_status": {
        "DNS": "Did Not Start",
        "DNF": "Did Not Finish",
        "DQ": "Disqualified",
        "NM": "No Mark (field events)",
        "NH": "No Height (vertical jumps)",
        "r": "Retired during race",
        "X": "Foul attempt"
    },

    "conditions": {
        "w": "Wind reading in m/s (after sprint/jump marks)",
        "(+)": "Wind-assisted (over 2.0 m/s, mark not valid for records)",
        "A": "Altitude (over 1000m, may affect performance)"
    },

    "competition_format": {
        "OW": "Olympic/World Championship level",
        "GL": "Gold Label / Diamond League",
        "Q": "Qualification round",
        "H": "Heat",
        "SF": "Semi-Final",
        "F": "Final"
    },

    "field_events": {
        "Auto Q": "Automatic qualifying mark (guarantees final spot)",
        "Q12/Q8": "Top 12 or 8 to final by position",
        "HS": "Height Series (high jump/pole vault)",
        "Runway": "Approach area for jumps/throws",
        "Toe Board": "Front edge of shot put/discus circle",
        "Scratch": "Foul attempt"
    }
}


###################################
# COACHING ANALYSIS FRAMEWORKS
###################################

ANALYSIS_FRAMEWORKS = {
    "performance_assessment": {
        "factors": [
            "Current form (last 3-5 performances)",
            "Season progression (improvement curve)",
            "PB age (how old is their best mark?)",
            "Championship experience",
            "Competition level quality"
        ]
    },

    "gap_analysis": {
        "to_medal": "Compare to historical medal marks at target championship",
        "to_final": "Compare to typical 8th place at target level",
        "to_semi": "Compare to cutoff for semi-finals",
        "to_pb": "How far from personal best?"
    },

    "form_indicators": {
        "improving": "Last 3 performances trending better",
        "peaking": "Hitting PB/SB at right time",
        "plateau": "Consistent but not improving",
        "declining": "Marks getting worse",
        "inconsistent": "High variance in performances"
    },

    "championship_readiness": {
        "factors": [
            "Rounds experience (heat → semi → final)",
            "Competition under pressure",
            "Major games history",
            "Current form trajectory",
            "Injury/health status"
        ]
    }
}


###################################
# HELPER FUNCTIONS
###################################

def get_event_knowledge(event: str) -> Dict[str, Any]:
    """Get comprehensive knowledge for a specific event."""
    event_lower = event.lower().replace(' ', '_').replace('-', '_')

    # Check each discipline
    if event_lower in ['100m', '200m', '400m']:
        return SPRINTS_KNOWLEDGE.get(event_lower, {})
    elif event_lower in ['800m', '1500m', 'mile']:
        return MIDDLE_DISTANCE_KNOWLEDGE.get(event_lower, {})
    elif event_lower in ['5000m', '10000m', 'marathon']:
        return LONG_DISTANCE_KNOWLEDGE.get(event_lower, {})
    elif 'hurdle' in event_lower or 'mh' in event_lower:
        if '110' in event_lower:
            return HURDLES_KNOWLEDGE.get('110mH_men', {})
        elif '100' in event_lower:
            return HURDLES_KNOWLEDGE.get('100mH_women', {})
        else:
            return HURDLES_KNOWLEDGE.get('400mH', {})
    elif 'high' in event_lower or 'hj' in event_lower:
        return JUMPS_KNOWLEDGE.get('high_jump', {})
    elif 'long' in event_lower and 'jump' in event_lower or 'lj' in event_lower:
        return JUMPS_KNOWLEDGE.get('long_jump', {})
    elif 'triple' in event_lower or 'tj' in event_lower:
        return JUMPS_KNOWLEDGE.get('triple_jump', {})
    elif 'pole' in event_lower or 'pv' in event_lower:
        return JUMPS_KNOWLEDGE.get('pole_vault', {})
    elif 'shot' in event_lower or 'sp' in event_lower:
        return THROWS_KNOWLEDGE.get('shot_put', {})
    elif 'discus' in event_lower or 'dt' in event_lower:
        return THROWS_KNOWLEDGE.get('discus', {})
    elif 'hammer' in event_lower or 'ht' in event_lower:
        return THROWS_KNOWLEDGE.get('hammer', {})
    elif 'javelin' in event_lower or 'jt' in event_lower:
        return THROWS_KNOWLEDGE.get('javelin', {})
    elif 'decathlon' in event_lower:
        return COMBINED_EVENTS_KNOWLEDGE.get('decathlon', {})
    elif 'heptathlon' in event_lower:
        return COMBINED_EVENTS_KNOWLEDGE.get('heptathlon', {})

    return {}


def get_discipline_category(event: str) -> str:
    """Categorize an event into its discipline."""
    event_lower = event.lower()

    if any(e in event_lower for e in ['100m', '200m', '400m']) and 'hurdle' not in event_lower:
        return 'sprints'
    elif any(e in event_lower for e in ['800m', '1500m', 'mile']):
        return 'middle_distance'
    elif any(e in event_lower for e in ['5000m', '10000m', '3000m', 'marathon', 'steeple']):
        return 'long_distance'
    elif 'hurdle' in event_lower or 'mh' in event_lower:
        return 'hurdles'
    elif any(e in event_lower for e in ['high', 'long', 'triple', 'pole', 'jump', 'vault']):
        return 'jumps'
    elif any(e in event_lower for e in ['shot', 'discus', 'hammer', 'javelin', 'throw', 'put']):
        return 'throws'
    elif any(e in event_lower for e in ['decathlon', 'heptathlon', 'combined']):
        return 'combined_events'

    return 'unknown'


def format_knowledge_for_context(event: str, include_benchmarks: bool = True) -> str:
    """Format event knowledge as context text for AI prompt."""
    knowledge = get_event_knowledge(event)

    if not knowledge:
        return f"No detailed knowledge available for {event}"

    parts = []

    # Overview
    if 'overview' in knowledge:
        parts.append(f"## {event.upper()} Overview\n{knowledge['overview']}")

    # Key phases or technique
    if 'phases' in knowledge:
        parts.append("\n### Race Phases")
        for phase, info in knowledge['phases'].items():
            if isinstance(info, dict):
                parts.append(f"- **{phase}**: {info.get('description', info.get('strategy', ''))}")

    if 'technique_components' in knowledge:
        parts.append("\n### Technique")
        for comp, info in knowledge['technique_components'].items():
            parts.append(f"- **{comp}**: {info.get('action', info.get('key_cue', ''))}")

    # Benchmarks
    if include_benchmarks and 'benchmarks' in knowledge:
        parts.append("\n### Performance Standards")
        for gender, marks in knowledge['benchmarks'].items():
            if isinstance(marks, dict):
                parts.append(f"\n**{gender.title()}:**")
                for level, mark in marks.items():
                    parts.append(f"- {level.replace('_', ' ').title()}: {mark}")

    return "\n".join(parts)


def get_all_terminology() -> Dict[str, Dict]:
    """Get all athletics terminology."""
    return ATHLETICS_TERMINOLOGY


# Export main knowledge bases
__all__ = [
    'SPRINTS_KNOWLEDGE',
    'MIDDLE_DISTANCE_KNOWLEDGE',
    'LONG_DISTANCE_KNOWLEDGE',
    'HURDLES_KNOWLEDGE',
    'JUMPS_KNOWLEDGE',
    'THROWS_KNOWLEDGE',
    'COMBINED_EVENTS_KNOWLEDGE',
    'ATHLETICS_TERMINOLOGY',
    'ANALYSIS_FRAMEWORKS',
    'get_event_knowledge',
    'get_discipline_category',
    'format_knowledge_for_context',
    'get_all_terminology'
]
