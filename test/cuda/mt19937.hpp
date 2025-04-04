#ifndef LCG_HPP
#define LCG_HPP

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <random>
#include <type_traits>

class Rng
{
protected:
    Rng()                 = default;
    Rng(const Rng& other) = default;

public:
    virtual ~Rng() = default;

public:
    virtual void gen(void* out, std::size_t len) = 0;

    template <typename T>
    T rand()
    {
        static_assert(
            std::is_standard_layout<T>::value && std::is_trivial<T>::value,
            "only support POD type");
        T r;
        this->gen((void*)(&r), sizeof(T));
        return r;
    }
};

class MT19937Rng : public Rng
{
private:
    std::mt19937_64 ctx;

public:
    MT19937Rng() noexcept
    {
    }

private:
    unsigned int next() noexcept
    {
        return (unsigned int)ctx();
    }

public:
    void reseed(unsigned int seed) noexcept
    {
        ctx.seed(seed);
    }

    void gen(void* out, std::size_t len) noexcept
    {
        std::uint8_t* data = (std::uint8_t*)out;
        while (len)
        {
            *data = this->next();
            data += 1, len -= 1;
        }
    }
};

#endif