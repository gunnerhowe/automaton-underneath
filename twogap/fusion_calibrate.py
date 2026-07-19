"""FUSION step 2 (per PREREG_FUSION frozen sequence): open ONLY calibration (m50 A/B/C/Bp, m10
A/B/C) + negatives (m0 A/B/C). Select anchor thresholds under the frozen constraints, compute the
multiplicative fraction + envelope, and the Amendment-1 fixed mid-grid anchor. m2 runs are NOT
readable by this script (allowlist below). Run: python fusion_calibrate.py"""
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parent
CAL = ["A_m50_s0", "B_m50_s0", "C_m50_s0", "Bp_m50_s0", "A_m10_s0", "B_m10_s0", "C_m10_s0"]
NEG = ["A_m0_s0", "B_m0_s0", "C_m0_s0"]
THETA_P = [round(0.30 + 0.05 * i, 2) for i in range(7)]      # 0.30..0.60
THETA_B = [0.55, 0.65, 0.75]
FIXED = (0.45, 0.65)                                          # Amendment-1 mid-grid anchor


def load(run):
    d = ROOT / "runs" / run
    probes = [json.loads(l) for l in (d / "probes.jsonl").read_text().splitlines()]
    log = json.loads((d / "log.json").read_text())
    return probes, log


def event_step(log):
    """First eval step with mean(toggle_8, dial_8) >= 0.9 for two consecutive evals."""
    seq = [(r["step"], (r["track_by"].get("toggle_8", 0) + r["track_by"].get("dial_8", 0)) / 2)
           for r in log]
    for i in range(len(seq) - 1):
        if seq[i][1] >= 0.9 and seq[i + 1][1] >= 0.9:
            return seq[i][0]
    return None


def anchor_step(probes, tp, tb):
    """First probe step with probe_mean >= tp AND shallow >= tb, sustained 2 consecutive points."""
    ok = [(p["step"], (p["probe_toggle"] + p["probe_dial"]) / 2 >= tp and p["shallow_acc"] >= tb)
          for p in probes]
    for i in range(len(ok) - 1):
        if ok[i][1] and ok[i + 1][1]:
            return ok[i][0]
    return None


def evaluate(tp, tb, data):
    leads, fracs, fa = [], [], 0
    ev_have_anchor = 0
    n_event = 0
    for run, (probes, log) in data.items():
        ev = event_step(log)
        an = anchor_step(probes, tp, tb)
        neg = run.split("_")[1] == "m0"
        if neg or ev is None:
            if an is not None:
                fa += 1
            continue
        n_event += 1
        if an is not None and an < ev:
            ev_have_anchor += 1
            leads.append(ev - an)
            fracs.append(an / ev)
    return dict(n_event=n_event, fired_pre_event=ev_have_anchor, fa=fa,
                med_lead=st.median(leads) if leads else 0, leads=leads, fracs=fracs)


def main():
    data = {r: load(r) for r in CAL + NEG}
    print("=== events (calibration + negatives) ===")
    for run in CAL + NEG:
        ev = event_step(data[run][1])
        print(f"  {run:<12} event={ev}")
    print("\n=== theta selection (constraints: fires pre-event on ALL eventing cal runs; 0 FA) ===")
    best = None
    for tp in THETA_P:
        for tb in THETA_B:
            r = evaluate(tp, tb, data)
            valid = r["n_event"] > 0 and r["fired_pre_event"] == r["n_event"] and r["fa"] == 0
            tag = "VALID" if valid else ""
            if valid and (best is None or r["med_lead"] > best[2]["med_lead"]):
                best = (tp, tb, r)
            print(f"  tp={tp:.2f} tb={tb:.2f}: eventing={r['n_event']} pre-event={r['fired_pre_event']} "
                  f"FA={r['fa']} med_lead={r['med_lead']:.0f} {tag}")
    print("\n=== Amendment-1 FIXED mid-grid anchor (0.45/0.65) ===")
    rf = evaluate(*FIXED, data)
    print(f"  eventing={rf['n_event']} pre-event={rf['fired_pre_event']} FA={rf['fa']} "
          f"med_lead={rf['med_lead']:.0f} fracs={[round(f,3) for f in rf['fracs']]}")
    out = dict(fixed=dict(tp=FIXED[0], tb=FIXED[1], fracs=rf["fracs"], leads=rf["leads"],
                          fa=rf["fa"], pre_event=rf["fired_pre_event"], n_event=rf["n_event"]))
    if best:
        tp, tb, r = best
        out["optimized"] = dict(tp=tp, tb=tb, fracs=r["fracs"], leads=r["leads"],
                                med_c=st.median(r["fracs"]), env=[min(r["fracs"]), max(r["fracs"])])
        print(f"\nSELECTED: tp={tp} tb={tb}  median_c={st.median(r['fracs']):.3f} "
              f"envelope=[{min(r['fracs']):.3f},{max(r['fracs']):.3f}] med_lead={r['med_lead']:.0f} "
              f"fracs={[round(f,3) for f in r['fracs']]}")
    else:
        print("\nKF1 CANDIDATE: no valid theta pair (check constraints)")
    if out["fixed"]["fracs"]:
        out["fixed"]["med_c"] = st.median(out["fixed"]["fracs"])
        out["fixed"]["env"] = [min(out["fixed"]["fracs"]), max(out["fixed"]["fracs"])]
    (ROOT / "fusion_constants.json").write_text(json.dumps(out, indent=1))
    print("\nwrote fusion_constants.json")


if __name__ == "__main__":
    main()
