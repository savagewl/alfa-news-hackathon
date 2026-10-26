from uuid import UUID, uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Company,
    CompanyMembership,
    CompanyPayment,
    CompanyPersonLink,
    CsvImport,
    Person,
    User,
)
from app.repositories.csv_import import lock_imports
from app.schemas.company import CreateCompanyRequest, PaymentDirection


class CompanyNotFoundError(LookupError):
    pass


class CompanyExistsError(ValueError):
    pass


class MemberUserNotFoundError(LookupError):
    pass


class CompanyRepository:
    def __init__(self, session: AsyncSession, user_id: UUID) -> None:
        self.session = session
        self.user_id = user_id

    def membership_filter(self):
        return (
            select(CompanyMembership.company_id)
            .where(
                CompanyMembership.company_id == Company.id,
                CompanyMembership.user_id == self.user_id,
            )
            .exists()
        )

    async def get(self, company_id: str) -> Company:
        company = await self.session.scalar(
            select(Company).where(Company.id == company_id, self.membership_filter())
        )
        if company is None:
            raise CompanyNotFoundError(company_id)
        return company

    async def list(
        self,
        *,
        limit: int,
        offset: int,
        q: str | None,
        group_id: str | None,
        region: str | None,
        is_placeholder: bool | None,
    ) -> dict:
        filters = [self.membership_filter()]
        if q:
            filters.append(
                or_(
                    Company.id.icontains(q, autoescape=True),
                    Company.name.icontains(q, autoescape=True),
                    Company.okved_name.icontains(q, autoescape=True),
                    Company.city.icontains(q, autoescape=True),
                )
            )
        if group_id is not None:
            filters.append(Company.group_id == group_id)
        if region is not None:
            filters.append(Company.region == region)
        if is_placeholder is not None:
            filters.append(Company.is_placeholder == is_placeholder)
        total = await self.session.scalar(select(func.count()).select_from(Company).where(*filters))
        items = await self.session.scalars(
            select(Company).where(*filters).order_by(Company.id).limit(limit).offset(offset)
        )
        return {"items": items.all(), "total": total, "limit": limit, "offset": offset}

    async def persons(self, company_id: str, *, limit: int, offset: int) -> dict:
        await self.get(company_id)
        condition = CompanyPersonLink.company_id == company_id
        total = await self.session.scalar(
            select(func.count()).select_from(CompanyPersonLink).where(condition)
        )
        rows = await self.session.execute(
            select(
                Person.id.label("person_id"),
                Person.city,
                Person.region,
                Person.last_import_id.label("person_import_id"),
                CompanyPersonLink.last_import_id.label("link_import_id"),
            )
            .join(CompanyPersonLink, CompanyPersonLink.person_id == Person.id)
            .where(condition)
            .order_by(Person.id)
            .limit(limit)
            .offset(offset)
        )
        return {"items": rows.mappings().all(), "total": total, "limit": limit, "offset": offset}

    async def create(self, body: CreateCompanyRequest) -> Company:
        # Share the import lock so a concurrent CSV cannot claim this ID first.
        await lock_imports(self.session)
        company_id = body.id or str(uuid4())
        if await self.session.scalar(select(Company.id).where(Company.id == company_id)):
            raise CompanyExistsError(company_id)
        company = Company(**{**body.model_dump(exclude={"id"}), "id": company_id})
        self.session.add(company)
        await self.session.flush()
        self.session.add(
            CompanyMembership(
                company_id=company_id,
                user_id=self.user_id,
                added_by_user_id=self.user_id,
            )
        )
        await self.session.commit()
        return company

    async def add_user(self, company_id: str, user_id: UUID) -> CompanyMembership:
        await self.get(company_id)
        if await self.session.get(User, user_id) is None:
            raise MemberUserNotFoundError(user_id)
        await self.session.execute(
            insert(CompanyMembership)
            .values(
                company_id=company_id,
                user_id=user_id,
                added_by_user_id=self.user_id,
            )
            .on_conflict_do_nothing(index_elements=["company_id", "user_id"])
        )
        membership = await self.session.scalar(
            select(CompanyMembership).where(
                CompanyMembership.company_id == company_id,
                CompanyMembership.user_id == user_id,
            )
        )
        await self.session.commit()
        return membership

    async def users(self, company_id: str, *, limit: int, offset: int) -> dict:
        await self.get(company_id)
        condition = CompanyMembership.company_id == company_id
        total = await self.session.scalar(
            select(func.count()).select_from(CompanyMembership).where(condition)
        )
        rows = await self.session.execute(
            select(
                User.user_id,
                User.full_name,
                CompanyMembership.created_at.label("joined_at"),
            )
            .join(CompanyMembership, CompanyMembership.user_id == User.user_id)
            .where(condition)
            .order_by(User.user_id)
            .limit(limit)
            .offset(offset)
        )
        return {"items": rows.mappings().all(), "total": total, "limit": limit, "offset": offset}

    @staticmethod
    def payment_condition(company_id: str, direction: PaymentDirection = "all"):
        incoming = CompanyPayment.beneficiary_company_id == company_id
        outgoing = CompanyPayment.payer_company_id == company_id
        return {"incoming": incoming, "outgoing": outgoing, "all": or_(incoming, outgoing)}[
            direction
        ]

    async def payments(
        self,
        company_id: str,
        *,
        limit: int,
        offset: int,
        direction: PaymentDirection,
        import_id: UUID | None,
    ) -> dict:
        await self.get(company_id)
        filters = [self.payment_condition(company_id, direction)]
        if import_id is not None:
            filters.append(CompanyPayment.import_id == import_id)
        total = await self.session.scalar(
            select(func.count()).select_from(CompanyPayment).where(*filters)
        )
        items = await self.session.scalars(
            select(CompanyPayment)
            .where(*filters)
            .order_by(CompanyPayment.id)
            .limit(limit)
            .offset(offset)
        )
        return {
            "items": items.all(),
            "total": total,
            "limit": limit,
            "offset": offset,
            "direction": direction,
            "import_id": import_id,
        }

    async def payment_imports(self, company_id: str, *, limit: int, offset: int) -> dict:
        await self.get(company_id)
        counts = (
            select(CompanyPayment.import_id, func.count().label("company_payment_rows"))
            .where(self.payment_condition(company_id))
            .group_by(CompanyPayment.import_id)
            .subquery()
        )
        total = await self.session.scalar(select(func.count()).select_from(counts))
        rows = await self.session.execute(
            select(
                CsvImport.id.label("import_id"),
                CsvImport.filename,
                CsvImport.created_at,
                CsvImport.row_count,
                counts.c.company_payment_rows,
            )
            .join(counts, counts.c.import_id == CsvImport.id)
            .order_by(CsvImport.created_at.desc(), CsvImport.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return {"items": rows.mappings().all(), "total": total, "limit": limit, "offset": offset}
