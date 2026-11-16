#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <zlib.h>

static uint64_t reverse_complement_code(uint64_t code, int k) {
    uint64_t rc = 0;

    for (int i = 0; i < k; ++i) {
        uint64_t b = code & 3ULL;
        code >>= 2;
        uint64_t cb = 3ULL - b;
        rc = (rc << 2) | cb;
    }

    return rc;
}

static int base_code(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default:  return -1;
    }
}

int main(int argc, char** argv) {

    std::string list_path;
    std::string out_path;
    int k = -1;

    for (int i = 1; i < argc; ++i) {

        std::string a = argv[i];

        if (a == "--k" && i + 1 < argc) {
            k = std::stoi(argv[++i]);
        }
        else if (a == "--list" && i + 1 < argc) {
            list_path = argv[++i];
        }
        else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        }
        else {
            std::cerr << "Unknown/incomplete argument: " << a << "\n";
            return 2;
        }
    }

    if (k <= 0 || k > 15 || list_path.empty() || out_path.empty()) {
        std::cerr << "Usage: --k K --list accessions.tsv --out matrix.raw\n";
        return 2;
    }

    const uint64_t space = 1ULL << (2 * k);
    const uint64_t mask = space - 1ULL;

    std::vector<uint8_t> canonical_flag(space, 0);

    for (uint64_t code = 0; code < space; ++code) {
        uint64_t rc = reverse_complement_code(code, k);
        uint64_t canon = std::min(code, rc);
        canonical_flag[canon] = 1;
    }

    std::vector<int32_t> canonical_index(space, -1);

    int32_t dim = 0;

    for (uint64_t code = 0; code < space; ++code) {
        if (canonical_flag[code]) {
            canonical_index[code] = dim++;
        }
    }

    std::vector<int32_t> code_to_index(space, -1);

    for (uint64_t code = 0; code < space; ++code) {
        uint64_t rc = reverse_complement_code(code, k);
        uint64_t canon = std::min(code, rc);

        if (canonical_index[canon] < 0) {
            throw std::runtime_error("canonical index construction failed");
        }

        code_to_index[code] = canonical_index[canon];
    }

    std::ifstream lf(list_path);

    if (!lf) {
        std::cerr << "Cannot open list: " << list_path << "\n";
        return 2;
    }

    std::vector<std::pair<std::string, std::string>> items;

    std::string line;

    while (std::getline(lf, line)) {

        if (line.empty()) {
            continue;
        }

        size_t tab = line.find('\t');

        if (tab == std::string::npos) {
            std::cerr << "Bad list line: " << line << "\n";
            return 2;
        }

        items.push_back({
            line.substr(0, tab),
            line.substr(tab + 1)
        });
    }

    std::ofstream out(out_path, std::ios::binary);

    if (!out) {
        std::cerr << "Cannot create output: " << out_path << "\n";
        return 2;
    }

    constexpr int BUF_SIZE = 1024 * 1024;
    std::vector<char> buf(BUF_SIZE);

    for (size_t item_i = 0; item_i < items.size(); ++item_i) {

        const std::string& accession = items[item_i].first;
        const std::string& fasta = items[item_i].second;

        gzFile gf = gzopen(fasta.c_str(), "rb");

        if (!gf) {
            std::cerr << "Cannot open FASTA with zlib: " << fasta << "\n";
            return 2;
        }

        gzbuffer(gf, BUF_SIZE);

        std::vector<uint64_t> counts(static_cast<size_t>(dim), 0ULL);

        uint64_t rolling = 0;
        int valid_run = 0;
        uint64_t valid_windows = 0;

        while (gzgets(gf, buf.data(), BUF_SIZE) != nullptr) {

            char* p = buf.data();

            if (*p == '>') {
                rolling = 0;
                valid_run = 0;
                continue;
            }

            for (; *p != '\0'; ++p) {

                char c = *p;

                if (c == '\n' || c == '\r') {
                    continue;
                }

                int b = base_code(c);

                if (b < 0) {
                    rolling = 0;
                    valid_run = 0;
                    continue;
                }

                rolling = ((rolling << 2) | static_cast<uint64_t>(b)) & mask;

                if (valid_run < k) {
                    ++valid_run;
                }

                if (valid_run >= k) {

                    int32_t idx = code_to_index[rolling];

                    if (idx < 0) {
                        std::cerr << "Internal canonical mapping error\n";
                        gzclose(gf);
                        return 2;
                    }

                    counts[static_cast<size_t>(idx)]++;
                    valid_windows++;
                }
            }
        }

        gzclose(gf);

        if (valid_windows == 0) {
            std::cerr
                << "No valid k-mers: accession="
                << accession
                << " fasta="
                << fasta
                << "\n";

            return 3;
        }

        std::vector<float> freq(static_cast<size_t>(dim), 0.0f);

        const double den = static_cast<double>(valid_windows);

        for (int32_t j = 0; j < dim; ++j) {
            freq[static_cast<size_t>(j)] =
                static_cast<float>(
                    static_cast<double>(counts[static_cast<size_t>(j)]) / den
                );
        }

        out.write(
            reinterpret_cast<const char*>(freq.data()),
            static_cast<std::streamsize>(
                freq.size() * sizeof(float)
            )
        );

        if (!out) {
            std::cerr << "Binary write failure\n";
            return 2;
        }

        if (
            item_i == 0 ||
            (item_i + 1) % 250 == 0 ||
            item_i + 1 == items.size()
        ) {
            std::cerr
                << "["
                << (item_i + 1)
                << "/"
                << items.size()
                << "] "
                << accession
                << " valid_windows="
                << valid_windows
                << "\n";
        }
    }

    out.close();

    std::cout
        << "K=" << k
        << "\tN=" << items.size()
        << "\tDIM=" << dim
        << "\tOUTPUT=" << out_path
        << "\n";

    return 0;
}
