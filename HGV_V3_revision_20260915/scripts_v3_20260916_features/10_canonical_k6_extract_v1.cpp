#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <zlib.h>

static inline int base_code(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

static inline uint16_t revcomp6(uint16_t x) {
    uint16_t rc = 0;
    for (int i = 0; i < 6; ++i) {
        uint16_t b = x & 3u;
        uint16_t cb = 3u - b;
        rc = static_cast<uint16_t>((rc << 2) | cb);
        x >>= 2;
    }
    return rc;
}

static std::vector<uint16_t> build_canonical_vocab() {
    std::vector<uint16_t> vocab;
    vocab.reserve(2080);

    for (uint16_t x = 0; x < 4096; ++x) {
        uint16_t rc = revcomp6(x);
        if (x <= rc) vocab.push_back(x);
    }

    if (vocab.size() != 2080) {
        throw std::runtime_error("Canonical k=6 vocabulary size is not 2080");
    }

    return vocab;
}

static std::array<int, 4096> build_code_to_index(
    const std::vector<uint16_t>& vocab
) {
    std::array<int, 4096> index{};
    index.fill(-1);

    for (size_t i = 0; i < vocab.size(); ++i) {
        index[vocab[i]] = static_cast<int>(i);
    }

    for (uint16_t x = 0; x < 4096; ++x) {
        uint16_t rc = revcomp6(x);
        uint16_t canon = std::min(x, rc);

        if (index[canon] < 0) {
            throw std::runtime_error("Canonical vocabulary mapping failure");
        }

        index[x] = index[canon];
    }

    return index;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " genome.fna.gz\n";
        return 2;
    }

    const std::string fasta_path = argv[1];

    std::vector<uint16_t> vocab = build_canonical_vocab();
    auto code_to_index = build_code_to_index(vocab);

    std::array<uint64_t, 2080> counts{};
    counts.fill(0);

    uint64_t total_sequence_bases = 0;
    uint64_t acgt_bases = 0;
    uint64_t ambiguous_bases = 0;
    uint64_t valid_kmers = 0;
    uint64_t contigs = 0;

    gzFile fp = gzopen(fasta_path.c_str(), "rb");
    if (!fp) {
        std::cerr << "ERROR: cannot open " << fasta_path << "\n";
        return 3;
    }

    const int BUFFER_SIZE = 1 << 20;
    std::vector<char> buffer(BUFFER_SIZE);

    uint16_t rolling = 0;
    int valid_run = 0;
    bool in_sequence = false;

    auto reset_contig = [&]() {
        rolling = 0;
        valid_run = 0;
    };

    while (gzgets(fp, buffer.data(), BUFFER_SIZE) != nullptr) {
        char* line = buffer.data();

        if (line[0] == '>') {
            ++contigs;
            in_sequence = true;
            reset_contig();
            continue;
        }

        if (!in_sequence) continue;

        for (char* p = line; *p; ++p) {
            char c = *p;

            if (c == '\n' || c == '\r' || c == ' ' || c == '\t') {
                continue;
            }

            ++total_sequence_bases;

            int b = base_code(c);

            if (b < 0) {
                ++ambiguous_bases;
                reset_contig();
                continue;
            }

            ++acgt_bases;

            rolling = static_cast<uint16_t>(
                ((rolling << 2) | static_cast<uint16_t>(b)) & 0x0FFFu
            );

            ++valid_run;

            if (valid_run >= 6) {
                int idx = code_to_index[rolling];

                if (idx < 0 || idx >= 2080) {
                    gzclose(fp);
                    std::cerr << "ERROR: invalid canonical index\n";
                    return 4;
                }

                ++counts[static_cast<size_t>(idx)];
                ++valid_kmers;
            }
        }
    }

    int gz_status = gzclose(fp);
    if (gz_status != Z_OK) {
        std::cerr << "ERROR: gzip close/check failure for "
                  << fasta_path << "\n";
        return 5;
    }

    if (contigs == 0) {
        std::cerr << "ERROR: no FASTA records found\n";
        return 6;
    }

    if (valid_kmers == 0) {
        std::cerr << "ERROR: zero valid canonical 6-mers\n";
        return 7;
    }

    double ambiguous_fraction =
        total_sequence_bases > 0
            ? static_cast<double>(ambiguous_bases) /
              static_cast<double>(total_sequence_bases)
            : 0.0;

    // First six fields are QC metadata.
    std::cout
        << contigs << '\t'
        << total_sequence_bases << '\t'
        << acgt_bases << '\t'
        << ambiguous_bases << '\t'
        << std::setprecision(12) << ambiguous_fraction << '\t'
        << valid_kmers;

    // Followed by exactly 2080 normalized canonical k-mer frequencies.
    std::cout << std::setprecision(10);

    for (size_t i = 0; i < counts.size(); ++i) {
        double f =
            static_cast<double>(counts[i]) /
            static_cast<double>(valid_kmers);

        std::cout << '\t' << f;
    }

    std::cout << '\n';

    return 0;
}
