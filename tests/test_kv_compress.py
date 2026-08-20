import unittest
from kv_compress.radix_cache import RadixPrefixTrie, MatchResult
from kv_compress.quantizer import AdaptiveKVQuantizer, QuantizedKVBlock
from kv_compress.telemetry import KVFinOpsTelemetry

class TestKVCompressX(unittest.TestCase):
    def setUp(self):
        self.trie = RadixPrefixTrie()
        # System prompt tokens: [101, 102, 103, 104, 105]
        self.system_prompt = [101, 102, 103, 104, 105]
        self.trie.insert(self.system_prompt, kv_block_handle="gpu_kv_block_0x99A")

    def test_radix_prefix_exact_and_partial_hit(self):
        # Query starting with exact system prompt + user question tokens
        user_query = [101, 102, 103, 104, 105, 999, 1000]
        match = self.trie.match_prefix(user_query)

        # Invariant: Matched first 5 tokens to pre-computed GPU memory handle
        self.assertEqual(match.prefix_length, 5)
        self.assertEqual(match.kv_block_handle, "gpu_kv_block_0x99A")
        self.assertEqual(match.matched_tokens, self.system_prompt)
        self.assertFalse(match.is_full_match)

    def test_adaptive_kv_quantization_compression_ratio(self):
        quantizer = AdaptiveKVQuantizer(sink_tokens=4, sliding_window_tokens=512)
        
        # 1. Short sequence: preserved in FP16 (ratio 1.0)
        short_res = quantizer.compress_sequence(total_seq_len=256)
        self.assertEqual(short_res.compression_ratio, 1.0)
        self.assertEqual(short_res.precision_tier, "FP16_SINK")

        # 2. Long 128k context sequence: deep history compressed to INT4
        long_res = quantizer.compress_sequence(total_seq_len=131072) # 128k tokens
        # Invariant: Compressed ratio must be < 0.28 (>= 72% VRAM reduction)
        self.assertLess(long_res.compression_ratio, 0.28)
        self.assertEqual(long_res.precision_tier, "INT4_HISTORY_HYBRID")

    def test_finops_telemetry_export(self):
        telemetry = KVFinOpsTelemetry(self.trie)
        # Perform 1 hit and 1 miss
        self.trie.match_prefix([101, 102, 103, 104, 105, 55])
        self.trie.match_prefix([999, 888])

        report = telemetry.export_finops_report(vram_saved_gb=142.5)
        self.assertEqual(report["radix_prefix_lookups"], 2)
        self.assertEqual(report["radix_prefix_hits"], 1)
        self.assertEqual(report["prefix_cache_hit_rate_pct"], 50.0)
        self.assertIn("a2z_soc_compliance_attestation", report)

if __name__ == "__main__":
    unittest.main()
