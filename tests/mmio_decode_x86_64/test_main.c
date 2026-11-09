#define _GNU_SOURCE 1
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mmio_decode_x86_64.h"

static int failures;
#define CHECK(name, expr) do { \
    if (!(expr)) { printf("FAIL: %s\n", name); failures++; } \
} while (0)

static struct {
    uint64_t value;
    uint32_t off;
    int size;
    int reads;
    int writes;
} device;

static uint64_t read_device(void *dev, uint32_t off, int size)
{
    (void)dev;
    device.off = off;
    device.size = size;
    device.reads++;
    return device.value;
}

static void write_device(void *dev, uint32_t off, uint64_t value, int size)
{
    (void)dev; (void)off; (void)value; (void)size;
    device.writes++;
}

static void reset(ucontext_t *uc, const uint8_t *code)
{
    memset(uc, 0, sizeof *uc);
    memset(&device, 0, sizeof device);
    MMIO_CTX_SET_PC(uc, (uintptr_t)code);
}

int main(void)
{
    /* DirectSound's test dword [base + index], 0x18 form. */
    const uint8_t test_sib[] = {
        0xF7, 0x04, 0x08, 0x18, 0x00, 0x00, 0x00, 0x0F
    };
    ucontext_t uc;
    uintptr_t start;

    reset(&uc, test_sib);
    start = MMIO_CTX_PC(&uc);
    device.value = 0x04;
    uc.uc_mcontext.gregs[REG_EFL] = 0x841; /* OF, ZF, CF */
    CHECK("F7 TEST handled", mmio_emulate_x86_64(
        &uc, 0x2000, NULL, read_device, write_device));
    CHECK("F7 TEST advances over ModRM, SIB and imm32",
          MMIO_CTX_PC(&uc) == start + 7);
    CHECK("F7 TEST reads the dword register once",
          device.reads == 1 && device.writes == 0 && device.size == 4);
    CHECK("F7 TEST preserves fault offset", device.off == 0x2000);
    CHECK("F7 TEST sets ZF and clears CF/OF",
          (uc.uc_mcontext.gregs[REG_EFL] & 0x841) == 0x40);

    /* TEST's sign flag follows the operand width for the byte form. */
    {
        const uint8_t test_byte[] = { 0xF6, 0x00, 0x80 };
        reset(&uc, test_byte);
        device.value = 0x80;
        CHECK("F6 TEST handled", mmio_emulate_x86_64(
            &uc, 0, NULL, read_device, write_device));
        CHECK("F6 TEST sets SF", (uc.uc_mcontext.gregs[REG_EFL] & 0x80) != 0);
        CHECK("F6 TEST uses byte width", device.size == 1);
    }

    /* Do not claim support for the arithmetic F7 groups. */
    {
        const uint8_t neg_mem[] = { 0xF7, 0x18 };
        reset(&uc, neg_mem);
        CHECK("F7 NEG remains unsupported", !mmio_emulate_x86_64(
            &uc, 0, NULL, read_device, write_device));
        CHECK("unsupported F7 does not touch the device", device.reads == 0);
        CHECK("unsupported F7 leaves RIP unchanged",
              MMIO_CTX_PC(&uc) == (uintptr_t)neg_mem);
    }

    if (failures) {
        printf("mmio_decode_x86_64: %d failure(s)\n", failures);
        return 1;
    }
    puts("mmio_decode_x86_64: ALL PASS");
    return 0;
}
