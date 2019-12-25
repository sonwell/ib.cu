#pragma once
#include "util/adaptor.h"
#include "util/getset.h"
#include "types.h"
#include "exceptions.h"

namespace cusparse {

enum class hyb_partition { automatic, user, maximum };
using hyb_partition_adaptor = util::adaptor<
	util::enum_container<hyb_partition_t,
			CUSPARSE_HYB_PARTITION_AUTO,
			CUSPARSE_HYB_PARTITION_USER,
			CUSPARSE_HYB_PARTITION_MAX>,
	util::enum_container<hyb_partition,
			hyb_partition::automatic,
			hyb_partition::user,
			hyb_partition::maximum>>;

inline void
create(hyb_mat_t& matrix)
{
	throw_if_error(cusparseCreateHybMat(&matrix));
}

inline void
destroy(hyb_mat_t& matrix)
{
	throw_if_error(cusparseDestroyHybMat(matrix));
}

class hyb_matrix : public cusparse::type_wrapper<hyb_mat_t> {
private:
	using base = cusparse::type_wrapper<hyb_mat_t>;
	using hp_a = hyb_partition_adaptor;
	hyb_partition _partition;

	hp_a hp() const { return _partition; }
	void hp(const hp_a& v) { _partition = v; }
public:
	util::getset<hp_a> partition;

	hyb_matrix() :
		base(), _partition(hyb_partition::automatic),
		partition([&] () { return hp(); }, [&] (const hp_a& v) { hp(v); }) {}
	explicit hyb_matrix(hyb_mat_t& matrix) :
		base(matrix), _partition(hyb_partition::automatic),
		partition([&] () { return hp(); }, [&] (const hp_a& v) { hp(v); }) {}
};

}
