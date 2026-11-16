#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static inline int base_code(char c) {
    switch(c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

static uint64_t revcomp_code(uint64_t x, int k) {
    uint64_t y = 0;
    for (int i=0;i<k;i++) {
        uint64_t b = x & 3ULL;
        y = (y << 2) | (3ULL - b);
        x >>= 2;
    }
    return y;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: canonical_kmer_one <fasta> <k> <out_float32_bin>\n";
        return 2;
    }
    const std::string fasta = argv[1];
    const int k = std::stoi(argv[2]);
    const std::string outp = argv[3];
    if (k < 1 || k > 12) {
        std::cerr << "unsupported k\n";
        return 2;
    }
    const uint64_t ncode = 1ULL << (2*k);
    std::vector<uint64_t> reps;
    reps.reserve(ncode/2 + 16);
    for (uint64_t x=0;x<ncode;x++) {
        uint64_t r = revcomp_code(x,k);
        if (x <= r) reps.push_back(x);
    }
    std::unordered_map<uint64_t,uint32_t> rep2idx;
    rep2idx.reserve(reps.size()*2);
    for (uint32_t i=0;i<reps.size();i++) rep2idx[reps[i]]=i;
    std::vector<uint32_t> code2idx(ncode);
    for (uint64_t x=0;x<ncode;x++) {
        uint64_t r=revcomp_code(x,k);
        uint64_t c=std::min(x,r);
        code2idx[x]=rep2idx[c];
    }
    std::vector<double> counts(reps.size(),0.0);
    std::ifstream in(fasta);
    if (!in) {
        std::cerr << "cannot open fasta: " << fasta << "\n";
        return 3;
    }
    const uint64_t mask = ncode - 1ULL;
    std::string line;
    uint64_t code=0, valid=0;
    int run=0;
    while (std::getline(in,line)) {
        if (!line.empty() && line[0]=='>') {
            code=0; run=0;
            continue;
        }
        for (char c: line) {
            int b=base_code(c);
            if (b<0) { code=0; run=0; continue; }
            code=((code<<2) | (uint64_t)b) & mask;
            run++;
            if (run>=k) {
                counts[code2idx[code]] += 1.0;
                valid++;
            }
        }
    }
    std::ofstream out(outp, std::ios::binary);
    if (!out) {
        std::cerr << "cannot open output: " << outp << "\n";
        return 4;
    }
    for (double x: counts) {
        float v = valid ? (float)(x/(double)valid) : 0.0f;
        out.write(reinterpret_cast<const char*>(&v), sizeof(float));
    }
    out.close();
    std::cout << "dim=" << reps.size() << " valid_kmers=" << valid << "\n";
    return 0;
}
