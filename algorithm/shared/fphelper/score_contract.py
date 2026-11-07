"""Completeness-aware research score output; never impute missing actions."""
from numbers import Integral
import json
from .data import ITEMS, LOW, HIGH, PHASE, totals

PHASES = ('rest', 'forehead_wrinkle', 'eye_closure', 'smile', 'snarl', 'lip_pucker')


def read_action_evidence(path, video_sha256, feature_protocol):
    if path is None:
        return [], None
    with open(path, encoding='utf-8') as stream:
        value = json.load(stream)
    phases = value.get('verified_phases')
    if (value.get('schema_version') != 1 or value.get('video_sha256') != video_sha256
            or value.get('feature_protocol') != feature_protocol
            or not isinstance(value.get('verification_method'), str)
            or not value['verification_method'].strip()
            or not isinstance(phases, list) or any(not isinstance(p, str) for p in phases)
            or len(set(phases)) != len(phases) or set(phases) - set(PHASES)):
        raise ValueError('Action evidence must match this video and feature-window protocol')
    return phases, value


def reviewed_result(legacy, verified_phases):
    result = dict(legacy)
    scores = score_result(legacy.get('subscores') or {}, verified_phases)
    result.update(scores)
    # Component totals from unverified predictions must not remain visible as confirmed scores.
    for name in ('rest_penalty', 'voluntary_score', 'synkinesis_penalty'):
        result.pop(name, None)
    result['reason'] = 'research_only_action_completeness_checked'
    result['requires_clinician_review'] = True
    if legacy.get('status') in ('failed', 'ungradable'):
        result['status'] = legacy['status']
        result['reason'] = 'feature_extraction_failed_or_ungradable'
    return result


def score_result(subscores, verified_phases):
    """verified_phases must come from confirmed action evidence, not time bins.

    An unverified action's supplied prediction is retained only as an unverified
    candidate. It cannot contribute to a complete Sunnybrook score.
    """
    if set(subscores) - set(ITEMS) or set(verified_phases) - set(PHASES):
        raise ValueError('Unknown subscore or action')
    valid, candidates, missing = {}, {}, []
    for j, name in enumerate(ITEMS):
        value = subscores.get(name)
        if value is None:
            missing.append(name)
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or not LOW[j] <= value <= HIGH[j]:
            raise ValueError('Subscore must be an integer within its Sunnybrook range')
        candidates[name] = int(value)
        if PHASES[PHASE[j]] in verified_phases:
            valid[name] = int(value)
    complete = len(valid) == len(ITEMS)
    raw = int(totals([valid[n] for n in ITEMS])) if complete else None
    return {'status': 'complete_research_score' if complete else 'partial_or_unverified',
            'subscores': valid, 'candidate_subscores': candidates,
            'missing_subscores': missing,
            'unverified_phases': [p for p in PHASES if p not in verified_phases],
            'raw_total': raw, 'total': raw if raw is not None and 0 <= raw <= 100 else None,
            'total_in_range': None if raw is None else 0 <= raw <= 100,
            'clinical_validation': False}
