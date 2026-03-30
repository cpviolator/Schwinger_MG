#include "mg_builder.h"

MGHierarchy build_full_mg(
    const DiracOp& D, const MGConfig& mcfg, const SolverConfig& scfg,
    std::mt19937& rng, int n_defl,
    bool verbose,
    const std::vector<Vec>* warm_start)
{
    int cb = mcfg.resolved_coarse_block();

    MGHierarchy mg;
    if (scfg.eigensolver == "feast") {
        auto feast_null = compute_near_null_space_feast(D, mcfg.k_null, scfg.feast_emax);
        mg = build_mg_hierarchy(D, mcfg.mg_levels, mcfg.block_size, mcfg.k_null,
                                cb, 0, rng, mcfg.w_cycle,
                                mcfg.pre_smooth, mcfg.post_smooth, verbose, &feast_null);
    } else {
        int null_iters = warm_start ? 5 : 20;
        mg = build_mg_hierarchy(D, mcfg.mg_levels, mcfg.block_size, mcfg.k_null,
                                cb, null_iters, rng, mcfg.w_cycle,
                                mcfg.pre_smooth, mcfg.post_smooth, verbose, warm_start);
    }

    if (n_defl > 0) {
        OpApply A = [&D](const Vec& s, Vec& d) { D.apply_DdagD(s, d); };
        mg.setup_sparse_coarse(A, D.lat.ndof, n_defl, 1e-12, 200,
                               scfg.eigensolver, scfg.feast_emax);
    }

    return mg;
}
