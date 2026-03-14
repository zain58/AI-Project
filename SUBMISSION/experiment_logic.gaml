model ScalingModel

global {
    int num_days    <- 60;
    int current_day <- 0;

    string scenario_id      <- "D1";
    int    repetition_id    <- 1;
    int    num_agents       <- 3000;
    string experiment_config <- "IC_ER";

    string input_folder  <- "C:/Users/moeez/Downloads/ATAI/projWeek/AI-Project/input_scaling/";
    string output_folder <- "C:/Users/moeez/Downloads/ATAI/projWeek/AI-Project/COMPR_output/";

    string run_folder   <- "";
    string voters_file  <- "";
    string edges_file   <- "";
    string results_file <- "";

    string net_type  <- "erdos_renyi";
    string pref_mod  <- "IC";

    float p_stubborn  <- 0.0;
    float p_strategic <- 0.0;
    float p_mixed     <- 0.0;

    list<string> candidates <- [
        "Macron", "Le Pen", "Melenchon", "Zemmour", "Pecresse",
        "Jadot", "Lassalle", "Roussel", "Dupont-Aignan",
        "Hidalgo", "Poutou", "Arthaud"
    ];

    map<string,int> scores <- map([]);
    list<string>    top2   <- [];

    float variance_val <- 0.0;
    float welfare_val  <- 0.0;
    int   changes_val  <- 0;

    action setup_metrics {
        scores <- map([]);
        loop c over: candidates { scores[c] <- 0; }
        ask voter {
            scores[current_vote] <- scores[current_vote] + 1;
        }
        list<string> sc <- candidates sort_by (- scores[each]);
        top2 <- length(sc) >= 2 ? [sc[0], sc[1]] : copy(sc);
        list<float> counts <- [];
        loop c over: candidates { counts <- counts + [float(scores[c])]; }
        float m <- mean(counts);
        float s <- 0.0;
        loop v over: counts { s <- s + (v - m)^2; }
        variance_val <- s / length(counts);
        changes_val <- length(voter where (each.changed));
        float w_sum <- 0.0;
        if length(top2) >= 1 {
            string c1 <- top2[0];
            string c2 <- length(top2) >= 2 ? top2[1] : c1;
            ask voter {
                int w1 <- self.get_welfare(c1);
                int w2 <- self.get_welfare(c2);
                w_sum <- w_sum + (c1 != c2 ? w1 + w2 : w1);
            }
        }
        welfare_val <- length(voter) > 0 ? w_sum / length(voter) : 0.0;
    }

    action save_daily {
        save [
            scenario_id, repetition_id, current_day, num_agents,
            net_type, pref_mod, p_stubborn, p_strategic, p_mixed,
            variance_val, welfare_val, changes_val
        ] to: results_file format: "csv";
    }

    init {
        if experiment_config = "IC_ER" { pref_mod <- "IC"; net_type <- "erdos_renyi"; }
        else if experiment_config = "IC_BA" { pref_mod <- "IC"; net_type <- "barabasi_albert"; }
        else if experiment_config = "Urn_ER" { pref_mod <- "Urn"; net_type <- "erdos_renyi"; }
        else if experiment_config = "Urn_BA" { pref_mod <- "Urn"; net_type <- "barabasi_albert"; }
        if scenario_id = "D1" { p_stubborn <- 0.6; p_strategic <- 0.2; p_mixed <- 0.2; }
        else if scenario_id = "D2" { p_stubborn <- 0.4; p_strategic <- 0.3; p_mixed <- 0.3; }
        else if scenario_id = "D3" { p_stubborn <- 0.3; p_strategic <- 0.4; p_mixed <- 0.3; }
        else if scenario_id = "D4" { p_stubborn <- 0.2; p_strategic <- 0.6; p_mixed <- 0.2; }
        else if scenario_id = "D5" { p_stubborn <- 0.2; p_strategic <- 0.3; p_mixed <- 0.5; }
        run_folder   <- input_folder + experiment_config + "/" + scenario_id + "_N" + string(num_agents) + "_run_001/";
        voters_file  <- run_folder + "voters.csv";
        edges_file   <- run_folder + "edges.csv";
        results_file <- output_folder + "simulation_results_" + experiment_config + "_" + scenario_id + "_N" + string(num_agents) + "_run_001.csv";
        matrix vm <- matrix(csv_file(voters_file, ",", true));
        loop i from: 0 to: vm.rows - 1 {
            create voter {
                voter_id   <- int(vm[0, i]);
                type       <- string(vm[1, i]);
                loyalty    <- float(vm[2, i]);
                prefs      <- [string(vm[3,i]), string(vm[4,i]), string(vm[5,i]), string(vm[6,i]),
                               string(vm[7,i]), string(vm[8,i]), string(vm[9,i]), string(vm[10,i]),
                               string(vm[11,i]), string(vm[12,i]), string(vm[13,i]), string(vm[14,i])];
                current_vote <- string(vm[15, i]);
                old_vote     <- current_vote;
                next_vote    <- current_vote;
            }
        }
        ask voter { neighbors <- []; }
        matrix em <- matrix(csv_file(edges_file, ",", true));
        loop i from: 0 to: em.rows - 1 {
            voter s <- one_of(voter where (each.voter_id = int(em[0, i])));
            voter t <- one_of(voter where (each.voter_id = int(em[1, i])));
            if s != nil and vt != nil {
                if !(t in s.neighbors) { add t to: s.neighbors; }
                if !(s in t.neighbors) { add s to: t.neighbors; }
            }
        }
        do setup_metrics;
        save ["scenario_id","repetition_id","day","num_agents","network_type","preference_model","prop_stubborn","prop_strategic","prop_mixed","variance_scores","social_welfare","num_changes"] 
        to: results_file format: "csv" rewrite: true;
        do save_daily;
    }

    reflex step when: current_day < num_days {
        current_day <- current_day + 1;
        ask voter { old_vote <- current_vote; }
        ask voter { do decide; }
        ask voter {
            current_vote <- next_vote;
            changed      <- (current_vote != old_vote);
        }
        do setup_metrics;
        do save_daily;
    }
}

species voter {
    int voter_id;
    string type;
    float loyalty;
    list<string> prefs;
    string current_vote;
    string old_vote;
    string next_vote;
    bool changed <- false;
    list<voter> neighbors <- [];
    map<string,int> local_poll <- map([]);
    list<string> local_top2 <- [];
    int get_rank(string c) {
        int i <- prefs index_of c;
        return i < 0 ? 12 : i + 1;
    }
    int get_welfare(string c) {
        int i <- prefs index_of c;
        return i < 0 ? 12 : i + 1;
    }
    action decide {
        local_poll <- map([]);
        loop c over: candidates { local_poll[c] <- 0; }
        loop n over: neighbors { local_poll[n.current_vote] <- local_poll[n.current_vote] + 1; }
        list<string> sc <- candidates sort_by (- local_results[each]);
        local_top2 <- length(sc) >= 2 ? [sc[0], sc[1]] : copy(sc);
        string fav <- prefs[0];
        string best_v <- local_top2[0];
        int r_best <- get_rank(best_v);
        loop v over: local_top2 { if get_rank(v) < r_best { best_v <- v; r_best <- get_rank(v); } }
        if type = "stubborn" { next_vote <- fav; }
        else if type = "strategic" { next_vote <- fav in local_top2 ? fav : best_v; }
        else {
            if fav in local_top2 { next_vote <- fav; }
            else {
                int tot <- length(neighbors);
                float p_fav <- tot > 0 ? float(local_poll[fav]) / tot : 0.0;
                float gap <- (tot > 0 and length(local_top2) >= 2) ? float(abs(local_poll[local_top2[0]] - local_poll[local_top2[1]])) / tot : 1.0;
                if (p_fav < 0.10) and (gap < 0.05) {
                    next_vote <- flip(1.0 - loyalty) ? best_v : fav;
                } else { next_vote <- fav; }
            }
        }
    }
}

experiment scaling_batch type: batch until: current_day = num_days {
    parameter "Config"   var: experiment_config among: ["IC_ER","IC_BA","Urn_ER","Urn_BA"];
    parameter "Scenario" var: scenario_id       among: ["D1","D2","D3","D4","D5"];
    parameter "Size"     var: num_agents        among: [3000, 5000];
}
